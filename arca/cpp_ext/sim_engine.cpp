/*
 * arca/cpp_ext/sim_engine.cpp
 * ===========================
 * Performance-critical simulation primitives exposed to Python via pybind11.
 *
 * Exposes:
 *   SimEngine.compute_reachability(adj_matrix) -> reachability_matrix
 *   SimEngine.batch_exploit(hosts, actions)    -> results vector
 *   SimEngine.floyd_warshall(adj)              -> shortest paths
 *
 * Build: pip install pybind11 && pip install -e ".[cpp]"
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <queue>
#include <limits>
#include <random>
#include <unordered_map>

namespace py = pybind11;

// ------------------------------------------------------------------ //
//  BFS-based reachability computation (faster than networkx for      //
//  dense adjacency on small graphs)                                  //
// ------------------------------------------------------------------ //
std::vector<std::vector<bool>> compute_reachability(
    const std::vector<std::vector<int>>& adj,
    int n_nodes
) {
    std::vector<std::vector<bool>> reach(n_nodes, std::vector<bool>(n_nodes, false));

    for (int src = 0; src < n_nodes; ++src) {
        std::vector<bool> visited(n_nodes, false);
        std::queue<int> q;
        q.push(src);
        visited[src] = true;
        reach[src][src] = true;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u < (int)adj.size()) {
                for (int v : adj[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        reach[src][v] = true;
                        q.push(v);
                    }
                }
            }
        }
    }
    return reach;
}

// ------------------------------------------------------------------ //
//  Floyd-Warshall for all-pairs shortest path                        //
// ------------------------------------------------------------------ //
std::vector<std::vector<double>> floyd_warshall(
    const std::vector<std::vector<double>>& weights,
    int n
) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dist(weights);

    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
    return dist;
}

// ------------------------------------------------------------------ //
//  Batch exploit simulation                                          //
//  Returns (success, reward) pairs for each action                   //
// ------------------------------------------------------------------ //
struct ExploitResult {
    bool success;
    double reward;
    int compromised_host;
};

std::vector<ExploitResult> batch_exploit(
    const std::vector<std::unordered_map<std::string, double>>& hosts,
    const std::vector<std::pair<int, int>>& actions,  // (target_host, exploit_id)
    uint64_t seed
) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<ExploitResult> results;
    results.reserve(actions.size());

    for (auto& [target, exploit_id] : actions) {
        if (target >= (int)hosts.size()) {
            results.push_back({false, -1.0, -1});
            continue;
        }
        auto& host = hosts[target];
        double prob = 0.5;  // default
        auto it = host.find("exploit_prob");
        if (it != host.end()) prob = it->second;

        bool success = dist(rng) < prob;
        double reward = success ? 20.0 : -0.5;
        results.push_back({success, reward, success ? target : -1});
    }
    return results;
}

// ------------------------------------------------------------------ //
//  pybind11 module                                                   //
// ------------------------------------------------------------------ //
PYBIND11_MODULE(_cpp_sim, m) {
    m.doc() = "ARCA C++ accelerated simulation engine";

    py::class_<ExploitResult>(m, "ExploitResult")
        .def_readonly("success", &ExploitResult::success)
        .def_readonly("reward", &ExploitResult::reward)
        .def_readonly("compromised_host", &ExploitResult::compromised_host);

    m.def("compute_reachability", &compute_reachability,
          py::arg("adj"), py::arg("n_nodes"),
          "BFS-based all-pairs reachability. Returns bool[n][n] matrix.");

    m.def("floyd_warshall", &floyd_warshall,
          py::arg("weights"), py::arg("n"),
          "All-pairs shortest path via Floyd-Warshall.");

    m.def("batch_exploit", &batch_exploit,
          py::arg("hosts"), py::arg("actions"), py::arg("seed") = 42ULL,
          "Batch exploit simulation. Returns list of ExploitResult.");

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__cpp_available__") = true;
}