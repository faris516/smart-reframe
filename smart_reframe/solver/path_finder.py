import numpy as np
import logging
from tqdm import tqdm

class PathSolver:
    """
    Offline Path Optimization using Dynamic Programming (Viterbi).
    Finds the optimal sequence of crop centers C = {x0, x1, ... xT} minimizing:
    Cost = sum( ContentLoss + SmoothnessCost + StaticBonus )
    """
    def __init__(self, frame_width: int, crop_width: int, transition_cost_factor: float = 0.005, static_bonus: float = 0.5):
        self.frame_width = frame_width
        self.crop_width = crop_width
        
        # Valid range for top-left corner x: [0, frame_width - crop_width]
        self.max_x = frame_width - crop_width
        # Discretize state space (1 pixel resolution is too slow? Maybe step=5)
        self.step = 2 # Pixels per state
        self.states = np.arange(0, self.max_x + 1, self.step)
        self.n_states = len(self.states)
        
        # Hyperparameters
        self.lambda_trans = transition_cost_factor # Penalty for moving
        self.lambda_static = static_bonus # Reward for staying EXACTLY the same
        
        logging.info(f"PathSolver initialized. States: {self.n_states} (Width: {frame_width}, Crop: {crop_width})")

    def solve(self, saliency_maps: list) -> list:
        """
        Runs Viterbi algorithm to find optimal path.
        Returns list of top-left X coordinates.
        """
        T = len(saliency_maps)
        if T == 0:
            return []
            
        logging.info(f"Solving optimal path for {T} frames...")
        
        # 1. Precompute Emission Costs (Content Loss)
        # Cost(x, t) = 1.0 - SaliencyCaptured(x, t)
        # We want to maximize Saliency, so we minimize (TotalPossibleSaliency - Captured)
        
        # Use numpy broadcasting for speed?
        # emission_matrix[t, state_idx]
        emission_cost = np.zeros((T, self.n_states))
        
        logging.info("Precomputing emission costs...")
        # Iterating per frame is unavoidable but inner loop can be vectorized?
        # saliency_maps is List[np.ndarray of size frame_width]
        
        # We can optimize: use integral images (cumsum) for O(1) window sum
        for t, smap in enumerate(tqdm(saliency_maps, desc="Cost Matrix")):
            # Compute integral image
            integral = np.cumsum(smap)
            # Prepend 0 for easier calc
            integral = np.insert(integral, 0, 0.0)
            
            # Vectorized window sum
            # Window starts: self.states
            # Window ends: self.states + self.crop_width
            x_starts = self.states
            x_ends = np.clip(x_starts + self.crop_width, 0, self.frame_width)
            
            # scores = integral[x_ends] - integral[x_starts]
            # Since integral is padded, indices are shifted? No, integral[i] is sum up to i-1.
            # Sum[start:end] = integral[end] - integral[start]
            scores = integral[x_ends] - integral[x_starts]
            
            # Normalize scores? Max possible score depends on frame content.
            # We want loss. 
            max_possible = scores.max() if scores.max() > 0 else 1.0
            
            # Cost = -Score (Minimization) or (Max - Score)
            emission_cost[t] = max_possible - scores
            
        # 2. Viterbi Forward Pass
        # dp[t, s] = min cost to reach state s at time t
        dp = np.full((T, self.n_states), np.inf)
        # parent[t, s] = index of state at t-1 that led to s
        parent = np.zeros((T, self.n_states), dtype=int)
        
        # Initial state (can start anywhere, equal prob? or prefer center?)
        # Prefer center start cost
        center_x = (self.max_x) // 2
        # Find closest state index to center
        # center_idx = np.abs(self.states - center_x).argmin()
        # dp[0, :] = emission_cost[0, :] # Start anywhere based on content
        
        # Add center start bias?
        start_dist = np.abs(self.states - center_x)
        dp[0] = emission_cost[0] + (start_dist * 0.001) 
        
        # Optimization: Restrict transition window
        # Camera can't jump from 0 to 1000 in one frame.
        # Max velocity constraint.
        MAX_VEL = 30 # pixels per frame
        MAX_VEL_STATES = int(MAX_VEL / self.step)
        
        logging.info("Running Viterbi forward pass...")
        
        for t in range(1, T):
            # For each state s at time t
            # dp[t, s] = emission[t,s] + min_over_k( dp[t-1, k] + trans_cost(k, s) )
            
            # This is O(N^2) per frame. With N=500, N^2=250k. T=1800 (60s). Total 450M ops.
            # Doable in Python? Iterative might be slow.
            # Vectorized approach:
            
            # Use a limited window around k to speed up?
            # Or just use Min-Convolution?
            
            prev_costs = dp[t-1]
            
            # This loop is the bottleneck.
            # Let's try a simplified approach: 
            # trans_cost(k, s) = lambda * (state[s] - state[k])^2
            
            # We iterate over current states 's'
            # But we only need to check 'k' within MAX_VEL of 's'
            
            # Optimization: 
            # Since smoothness is quadratic, we can use distance transform?
            # For now, let's just loop with window.
            
            for s_idx in range(self.n_states):
                # Search window in previous states
                k_min = max(0, s_idx - MAX_VEL_STATES)
                k_max = min(self.n_states, s_idx + MAX_VEL_STATES + 1)
                
                # Previous costs window
                w_prev = prev_costs[k_min:k_max]
                w_states = self.states[k_min:k_max]
                curr_pos = self.states[s_idx]
                
                # Calculate movement costs
                # L2 Squared
                move_costs = self.lambda_trans * ((w_states - curr_pos)**2)
                
                # Static Bonus (Reward 0 movement)
                # If w_states == curr_pos (index match), subtract bonus (reduce cost)
                # dists = abs(w_states - curr_pos)
                # bonus_mask = (dists < 0.1)
                # move_costs[bonus_mask] -= self.lambda_static
                
                total_k = w_prev + move_costs
                
                best_k_idx = np.argmin(total_k)
                min_val = total_k[best_k_idx]
                
                dp[t, s_idx] = emission_cost[t, s_idx] + min_val
                parent[t, s_idx] = k_min + best_k_idx

        # 3. Backtrace
        logging.info("Backtracing optimal path...")
        path_indices = np.zeros(T, dtype=int)
        
        # End state: min cost at T-1
        path_indices[-1] = np.argmin(dp[-1])
        
        for t in range(T-2, -1, -1):
            path_indices[t] = parent[t+1, path_indices[t+1]]
            
        optimal_path = self.states[path_indices]
        return optimal_path.tolist()
