"""C++ extension module. Falls back gracefully if not compiled."""

try:
    from arca._cpp_sim import compute_reachability, floyd_warshall, batch_exploit  # type: ignore
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

    def compute_reachability(adj, n_nodes):
        """Pure-Python fallback."""
        from collections import deque
        reach = [[False] * n_nodes for _ in range(n_nodes)]
        for src in range(n_nodes):
            visited = [False] * n_nodes
            q = deque([src])
            visited[src] = True
            reach[src][src] = True
            while q:
                u = q.popleft()
                for v in (adj[u] if u < len(adj) else []):
                    if not visited[v]:
                        visited[v] = True
                        reach[src][v] = True
                        q.append(v)
        return reach

    def floyd_warshall(weights, n):
        """Pure-Python fallback."""
        import math
        dist = [row[:] for row in weights]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def batch_exploit(hosts, actions, seed=42):
        import random
        rng = random.Random(seed)
        results = []
        for target, exploit_id in actions:
            if target >= len(hosts):
                results.append({"success": False, "reward": -1.0, "compromised_host": -1})
                continue
            prob = hosts[target].get("exploit_prob", 0.5)
            success = rng.random() < prob
            results.append({
                "success": success,
                "reward": 20.0 if success else -0.5,
                "compromised_host": target if success else -1,
            })
        return results


__all__ = ["CPP_AVAILABLE", "compute_reachability", "floyd_warshall", "batch_exploit"]