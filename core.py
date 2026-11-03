import numpy as np

EPSILON = 1e-6
BASE_CONVERSION = 0.8


def distance(dim, p=2):
    result = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            result[i,j] = np.abs(i/dim[0] - j/dim[1])**p
    return result


def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))


class Graph:
    def __init__(self, source, sink, cost=None, gammas=None):
        self.source = np.asarray(source, dtype=float)
        self.sink = np.asarray(sink, dtype=float)
        # self.source /= np.sum(self.source)
        # self.sink /= np.sum(self.sink)
        assert np.allclose(np.sum(source), np.sum(sink)), (np.sum(source), np.sum(sink))
        self.dim = (len(self.source), len(self.sink))
        cost = cost if cost is not None else distance(self.dim)
        self.cost = np.asarray(cost)
        gammas = gammas if gammas is not None else np.full(self.dim[0], 0.01)
        self.gammas = np.asarray(gammas)
        self.transport = np.zeros(self.dim)

    def solve_uniform(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.transport[i,j] = self.source[i] * self.sink[j] / self.sink.sum()
        history = [self.objective()]
        return self.transport, history

    def update_transport(self, u, k, v, rho=None):
        if rho is None:
            rho = np.ones(self.dim[0])
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.transport[i,j] = u[i] * k[i,j] * (v[j] ** rho[i])

    def sinkhorn(self, max_iter=10000):
        gamma_bar = np.mean(self.gammas)
        u = np.ones(self.dim[0])
        v = np.ones(self.dim[1])
        k = np.exp(-self.cost / gamma_bar)
        # k = np.ones(self.dim) # if gamma_bar == math.inf
        # k = np.eye(*self.dim) # if gamma_bar == 0 # unsolvable
        history = []
        for _ in range(max_iter):
            for i in range(self.dim[0]):
                u[i] = self.source[i] / (k[i,:] @ v)
            for j in range(self.dim[1]):
                v[j] = self.sink[j] / (k[:,j] @ u)
            self.update_transport(u, k, v)
            history.append(self.objective())
            if self.verify_constraints():
                break
        return self.transport, history

    def general_sinkhorn(self, max_iter=10000):
        gamma_bar = np.mean(self.gammas)
        rho = gamma_bar / self.gammas
        u = np.ones(self.dim[0])
        w = np.ones(self.dim[1])
        k = np.ones(self.dim)
        history = []
        for i in range(self.dim[0]):
            k[i,:] = np.exp(-self.cost[i,:] / self.gammas[i])        
        for _ in range(max_iter):
            for i in range(self.dim[0]):
                u[i] = self.source[i] / (k[i,:] @ (w ** rho[i]))
            self.update_transport(u, k, w, rho)
            for j in range(self.dim[1]):
                tj = self.transport[:,j]
                w[j] *= 1 - (np.sum(tj) - self.sink[j]) / (tj @ rho + EPSILON)
                w[j] = max(w[j], 0)
            self.update_transport(u, k, w, rho)
            obj = self.objective()
            history.append(obj)
            # print(obj)
            if self.verify_constraints() or np.isnan(obj):
                break
        return self.transport, history

    def total_cost(self):
        return np.sum(self.cost * self.transport)

    def objective(self):
        main_obj = self.total_cost()
        reg_obj = np.zeros(self.dim)
        non_zeros = self.transport[self.transport > 0]
        reg_obj[self.transport > 0] = non_zeros * np.log(non_zeros)
        result = main_obj + np.sum(reg_obj, axis=1) @ self.gammas
        return result.item()

    def verify_constraints(self):
        c1 = np.allclose(self.transport.sum(0), self.sink)
        c2 = np.allclose(self.transport.sum(1), self.source)
        return c1 and c2


class GraphPricing(Graph):
    """
    session = intent = NSP5M
    total_demand = session * base_conversion = min_demand + flex_demand
    edge_cost = P(complete | request, eta) if not slack
    transport = conversion = match
    conversion_target = P(request | intent, pt)
    objective = conversion + gamma * smoothness
    rides = mached and completed
    primetime = f(conversion_target)
    cancel = converted min_demand and not completed
    shed = not converted flex_demand given pt
    converted = matched or canceled
    """
    def __init__(self, session, supply, hvm_supply=None, hvm_ratio=1.0):
        self.session = np.asarray(session, dtype=float)
        self.supply = np.asarray(supply, dtype=float)
        hvm_supply = hvm_supply if hvm_supply is not None else np.zeros_like(self.supply)
        self.hvm_supply = np.asarray(hvm_supply, dtype=float)
        self.hvm_ratio = hvm_ratio
        self.el = ElasticityLearner(base=BASE_CONVERSION, elasticity=1)
        self.gh_dim = (len(self.session), len(self.supply))
        self.db_demand_ratio = 0.6

        self.demand_types = ["min", "flex_d1", "flex_d2", "hvm_min", "hvm_flex", "slack"]
        self.supply_types = ["standard", "hvm", "slack"]
        self.demand_nodes = {}
        self.supply_nodes = {}
        self.edge_costs = {}
        self.gammas_db = {}

        source, sink, cost, gammas = self.presolve()
        cost = -cost
        super().__init__(source, sink, cost=cost, gammas=gammas)

    def presolve(self):
        gamma = 0.01
        gamma_d2 = gamma * 2.0
        max_pt = 5.0
        p, k, x0 = 1.0, -5.0, 0.2
        cancel_penalty = -0.5
        unconnected_edge_cost = -1.0
        cmd = True

        total_demand = self.inverted_demand(0)
        min_demand = self.inverted_demand(max_pt)
        flex_demand = total_demand - min_demand
        total_supply = self.supply + self.hvm_supply
        standard_flex_demand = flex_demand * self.hvm_ratio
        self.demand_nodes["min"] = min_demand * self.hvm_ratio
        self.demand_nodes["flex_d1"] = standard_flex_demand * self.db_demand_ratio
        self.demand_nodes["flex_d2"] = standard_flex_demand * (1 - self.db_demand_ratio)
        self.demand_nodes["hvm_min"] = min_demand * (1 - self.hvm_ratio)
        self.demand_nodes["hvm_flex"] = flex_demand * (1 - self.hvm_ratio)
        self.demand_nodes["slack"] = np.array([np.sum(total_supply)])
        self.supply_nodes["standard"] = self.supply
        self.supply_nodes["hvm"] = self.hvm_supply
        self.supply_nodes["slack"] = np.array([np.sum(total_demand)])

        gh_d, gh_s = self.gh_dim
        for dt in self.demand_types:
            dd = 1 if dt == "slack" else gh_d
            for st in self.supply_types:
                sd = 1 if st == "slack" else gh_s
                self.edge_costs[(dt, st)] = np.full((dd, sd), unconnected_edge_cost)
            self.gammas_db[dt] = np.full(dd, gamma)
        self.gammas_db["flex_d2"].fill(gamma_d2)

        edge_cost = self.init_edge_cost(p, k, x0)
        for t in ["min", "flex_d1", "flex_d2"]:
            self.edge_costs[(t, "standard")] = edge_cost
            if cmd:
                self.edge_costs[(t, "hvm")] = edge_cost
        for t in ["hvm_min", "hvm_flex"]:
            self.edge_costs[(t, "hvm")] = edge_cost
        self.edge_costs[("min", "slack")].fill(cancel_penalty)
        self.edge_costs[("flex_d1", "slack")].fill(-gamma)
        self.edge_costs[("flex_d2", "slack")].fill(-gamma)
        self.edge_costs[("hvm_min", "slack")].fill(cancel_penalty)
        self.edge_costs[("hvm_flex", "slack")].fill(-gamma)
        for st in self.supply_types:
            self.edge_costs[("slack", st)].fill(0)

        source = np.concatenate([self.demand_nodes[t] for t in self.demand_types])
        sink = np.concatenate([self.supply_nodes[t] for t in self.supply_types])
        gammas = np.concatenate([self.gammas_db[t] for t in self.demand_types])
        rows = []
        for dt in self.demand_types:
            row = [self.edge_costs[(dt, st)] for st in self.supply_types]
            rows.append(np.concatenate(row, axis=1))
        cost = np.concatenate(rows, axis=0)
        return source, sink, cost, gammas

    def total_demand(self):
        return self.session * self.el.base

    def inverted_demand(self, pt):
        conversion = self.el.conversion(pt)
        return self.session * conversion

    def init_edge_cost(self, p, k, x0):
        eta = distance(self.gh_dim, p)
        return logistic(eta, k, x0)

    def solve(self):
        return self.general_sinkhorn()

    def split_transport(self):
        result = {}
        di = 0
        gh_d, gh_s = self.gh_dim
        for dt in self.demand_types:
            dd = 1 if dt == "slack" else gh_d
            si = 0
            for st in self.supply_types:
                sd = 1 if st == "slack" else gh_s
                result[(dt, st)] = self.transport[di:di+dd,si:si+sd]
                si += sd
            di += dd
        return result

    def split_requests(self):
        result = {}
        split_transport = self.split_transport()
        for (dt, st), transport in split_transport.items():
            result[(dt, st)] = np.sum(transport, axis=1)
        return result

    def converted_requests(self):
        requests = self.split_requests()
        pairs = {
            "min": [
                ("min", "slack"),
                ("min", "standard"),
                ("min", "hvm"),
            ],
            "flex_d1": [
                ("flex_d1", "standard"),
                ("flex_d1", "hvm"),
            ],
            "flex_d2": [
                ("flex_d2", "standard"),
                ("flex_d2", "hvm"),
            ],
            "hvm": [
                ("hvm_min", "slack"),
                ("hvm_min", "hvm"),
                ("hvm_flex", "hvm"),
            ],
        }
        result = {}
        for k in pairs:
            result[k] = np.sum([requests[p] for p in pairs[k]], axis=0)
        return result

    def conversion_target(self):
        converted = self.converted_requests()
        converted_db = {}
        converted_db["standard"] = converted["min"] + converted["flex_d1"] + converted["flex_d2"]
        converted_db["d1"] = converted["min"] * self.db_demand_ratio + converted["flex_d1"]
        converted_db["d2"] = converted["min"] * (1 - self.db_demand_ratio) + converted["flex_d2"]
        converted_db["hvm"] = converted["hvm"]
        session_db = {}
        session_db["standard"] = self.session * self.hvm_ratio
        session_db["d1"] = self.session * self.hvm_ratio * self.db_demand_ratio
        session_db["d2"] = self.session * self.hvm_ratio * (1 - self.db_demand_ratio)
        session_db["hvm"] = self.session * (1 - self.hvm_ratio)
        result = {}
        for k in converted_db:
            result[k] = converted_db[k] / (session_db[k] + EPSILON)
        return result

    def primetime(self):
        cvr = self.conversion_target()
        return {k : self.el.pt(cvr[k]) for k in cvr}


class ElasticityLearner:
    def __init__(self, base, elasticity):
        self.base = base
        self.elasticity = elasticity

    def conversion(self, pt):
        return self.base * (1 + pt) ** -self.elasticity
    
    def pt(self, conversion):
        if np.sum(conversion) == 0:
            return np.zeros_like(conversion)
        return (conversion / self.base) ** (1 / -self.elasticity) - 1


class ConversionLearner:
    def __init__(self, alpha=2, pi=-0.5, tao=-0.5):
        self.alpha = alpha
        self.pi = pi
        self.tao = tao


class DPO:
    def __init__(self, cl=None, etas=None, mode_mix=None, base_prices=None, base_conversion=BASE_CONVERSION, mu=100):
        self.cl = cl if cl else ConversionLearner()
        self.etas = etas if etas else [0, 5.7, 20]
        self.mode_mix = mode_mix if mode_mix else [0.1, 0.6, 0.3]
        self.base_prices = base_prices if base_prices else [24, 21.6, 19.6]
        self.base_conversion = base_conversion
        self.mu = mu

        self.dim = len(self.etas)
        self.etas = np.asarray(self.etas, dtype=float)
        self.mode_mix = np.asarray(self.mode_mix, dtype=float)
        self.base_prices = np.asarray(self.base_prices, dtype=float)
        self.mode_mix /= np.sum(self.mode_mix)
        self.thetas = self.init_thetas()
        # self.thetas = np.array([0, 0, 0])
        self.theta0 = self.init_theta0()

    def init_thetas(self):
        result = np.zeros(self.dim)
        if len(result) == 1:
            return result
        result[1] = 0
        m0, b0, t0 = self.mode_mix[1], self.base_prices[1], self.etas[1]
        for i in range(self.dim):
            m, b, t = self.mode_mix[i], self.base_prices[i], self.etas[i]
            theta = np.log(m/m0) - self.cl.pi * (b-b0) - self.cl.tao * (t-t0)
            result[i] = theta
        return result
    
    def init_theta0(self):
        b = self.base_conversion
        u = self.utilities(self.base_prices)
        return np.log((1 - b) / b * np.sum(np.exp(u)))

    def utilities(self, prices):
        result = np.zeros(self.dim)
        for i, (theta, p, t) in enumerate(zip(self.thetas, prices, self.etas)):
            u = theta + self.cl.alpha + self.cl.pi * p + self.cl.tao * t
            result[i] = u
        return result

    def conversion(self, prices):
        assert len(prices) == len(self.etas)
        utilities = self.utilities(prices)
        den = np.exp(self.theta0) + np.sum(np.exp(utilities))
        return np.exp(utilities) / den

    def booking(self, prices):
        cvrs = self.conversion(prices)
        bookings = [cvr * p for p, cvr in zip(prices, cvrs)]
        return np.sum(bookings)

    def constraint(self, prices, ct):
        cvrs = self.conversion(prices)
        return abs(np.sum(cvrs) - ct)
        
    def objective(self, prices, ct):
        penalty = self.constraint(prices, ct) ** 2
        objective = self.booking(prices) - self.mu * penalty
        return objective

    def gradient(self, prices, ct):
        result = np.zeros(self.dim)
        cvrs = self.conversion(prices)
        for i in range(self.dim):
            part_der = cvrs[i]
            penalty_der = 0
            for j in range(self.dim):
                if i == j:
                    cvr_der = cvrs[j] * (1 - cvrs[i])
                else:
                    cvr_der = -1 * cvrs[j] * cvrs[i]
                cvr_der = cvr_der * self.cl.pi
                part_der += prices[j] * cvr_der
                penalty_der += cvr_der
            penalty_der *= 2 * (np.sum(cvrs) - ct)
            result[i] = part_der - self.mu * penalty_der
        return result

    def numerical_gradient(self, prices, ct, eps=0.0001):
        result = np.zeros(self.dim)
        for i in range(len(prices)):
            p = prices.copy()
            p[i] = prices[i] + eps
            right = self.objective(p, ct)
            p[i] = prices[i] - eps
            left = self.objective(p, ct)
            result[i] = (right - left) / (2 * eps)
        return result

    def gradient_descent(self, ct, max_iter=1000, lr=0.1, threshold=0.1):
        history = []
        prices = np.zeros(self.dim)
        for i in range(max_iter):
            grad = self.gradient(prices, ct)
            # grad = self.numerical_gradient(prices, ct)
            prices += lr * grad
            obj = self.objective(prices, ct)
            history.append(obj)
            if np.linalg.norm(grad) <= threshold:
                break
        return prices, history

    def solve(self, cvrs):
        result = []
        for cvr in cvrs:
            price, history = self.gradient_descent(cvr)
            result.append(price)
        return result

    def invert(self, ct):
        utility = np.log(1 / (1 - ct) - 1)
        return (utility - self.cl.alpha - self.cl.tao * self.etas) / self.cl.pi

    def derivative(self, price, ct):
        cvr = self.conversion(price)
        grad_cvr = cvr * (1 - cvr) * self.cl.pi
        grad_booking = cvr + price * grad_cvr
        grad_penalty = -2 * self.mu * (cvr - ct) * grad_cvr
        return grad_booking + grad_penalty
