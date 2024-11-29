import numpy as np
import typing as t

class BioHoloneticModel:
    def __init__(
        self, 
        n_dimensions: int, 
        n_clusters: int,
        noise_sigma: float = 0.1
    ):
        self.n_dimensions = n_dimensions
        self.n_clusters = n_clusters
        self.noise_sigma = noise_sigma
        
        self.state = np.zeros(n_dimensions)
        self.prev_state = np.zeros(n_dimensions)
        
        self.goals = np.array([
            [1, 2, 3],
            [-1, -2, -3],
            [0.5, 0.5, 0.5]
        ])
        
        self.clusters_positive = self._input_clusters()
        self.clusters_negative = -self.clusters_positive
        
        self.cue_generator = np.random.RandomState()
        self.goal = self.goals[0]  # Start with the first goal
        self.stop_counter = 0

        # Get the regulation factor and stop threshold during initialization
        self.regulation_factor = self._get_valid_input(
            "Please enter the regulation factor (a positive number): ", 
            float, 
            lambda x: x > 0
        )
        self.stop_threshold = self._get_valid_input(
            "Please enter the stop threshold (a positive integer): ", 
            int, 
            lambda x: x > 0
        )

    def _input_clusters(self) -> np.ndarray:
        clusters = []
        for i in range(self.n_clusters):
            while True:
                values = input(f"Cluster {i+1}, please enter three values between 0 and 1, separated by spaces: ")
                try:
                    cluster_values = [float(v) for v in values.split()]
                    if len(cluster_values) == 3 and all(0 <= v <= 1 for v in cluster_values):
                        clusters.append(cluster_values)
                        break
                    else:
                        print("Non-acceptable values. Please try again.")
                except ValueError:
                    print("Invalid input format. Please enter three numerical values.")
        return np.array(clusters)

    def compute_delta_state(self, external_force: np.ndarray) -> np.ndarray:
        delta = self.goal - self.state
        print(f"Delta: {delta}")

        if np.any(np.abs(delta) > 3):
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        cluster_contributions = np.zeros(self.n_dimensions)

        for i in range(self.n_dimensions):
            selected_cluster = None
            if np.abs(delta[i]) > 3:
                # Determine the selected cluster based on delta and goal
                if np.sign(self.goal[i]) == 1:
                    if delta[i] > 0:
                        selected_cluster = self.clusters_positive[:, i]
                    else:
                        selected_cluster = self.clusters_negative[:, i]
                else:
                    if delta[i] < 0:
                        selected_cluster = self.clusters_negative[:, i]
                    else:
                        selected_cluster = self.clusters_positive[:, i]

                # Compute the regulation factor
                Di = self._compute_dynamic_regulation(delta, i)
                regulated_cluster = Di * selected_cluster                # Compute the regulation factor
                Di = self._compute_dynamic_regulation(delta, i)
                regulated_cluster = Di * selected_cluster
                print(f"Dimension {i}: Regulated Cluster = {regulated_cluster}, Total Contribution = {cluster_contributions[i]}")
            else:
                # When no regulation is applied, we add all cluster contributions
                if delta[i] >= 0:
                    selected_cluster = self.clusters_positive[:, i]
                else:
                    selected_cluster = self.clusters_negative[:, i]
                regulated_cluster = selected_cluster
                print("No regulation is applied.")

            # Add the regulated cluster to contributions
            cluster_contributions[i] += np.sum(regulated_cluster)

            # Add contributions from all other clusters
            for j in range(self.n_clusters):
                if not np.all(self.clusters_positive[j, i] == selected_cluster):
                    cluster_contributions[i] += self.clusters_positive[j, i]
                if not np.all(self.clusters_negative[j, i] == selected_cluster):
                    cluster_contributions[i] += self.clusters_negative[j, i]

        noise = np.random.normal(0, self.noise_sigma, self.n_dimensions)
        external_force_with_noise = external_force + noise

        delta_state = cluster_contributions + external_force_with_noise
        return delta_state
    
    def _compute_dynamic_regulation(self, delta: np.ndarray, cluster_idx: int) -> float:
        sigmoid_activation = lambda x: 1 / (1 + np.exp(-x))

        # Select the appropriate cluster based on the delta value
        if delta[cluster_idx] >= 0:
            selected_cluster = self.clusters_positive[:, cluster_idx]
        else:
            selected_cluster = -self.clusters_negative[:, cluster_idx]

        # Compute the initial regulation factor as the sum of the absolute value of the selected cluster
        regulation_factor = np.sum(np.abs(selected_cluster))

        # Apply sigmoid activation to the regulation factor
        sigmoid_value = sigmoid_activation(regulation_factor * 10)

        # Combine the regulation factor with the user-provided value
        final_regulation_factor = self.regulation_factor * sigmoid_value

        # Ensure the final regulation factor is at least the minimum value
        return max(final_regulation_factor, 1.1)

    def trigger_goal_transition(
        self, 
        external_input: t.Optional[np.ndarray] = None
    ) -> np.ndarray:
        cue = self.cue_generator.randint(1, 6)
        if cue == 5:
            print("Cue-based trigger activated. Switching to the second goal.")
            return self.goals[1]

        transition_mask = self._identify_transition_regions()
        if np.any(transition_mask):
            print("State-based trigger activated. Switching to the first goal.")
            return self.goals[0]

        if self.stop_counter >= 3:
            print("Out-of-range trigger activated. Switching to the third goal.")
            return self.goals[2]

        print("No trigger activated. Retaining current goal.")
        return self.goal

    def _identify_transition_regions(self) -> np.ndarray:
        transition_mask = (
            (self.state > 0.8) | 
            (self.state < -0.8) | 
            (np.abs(self.state) < 0.1)
        )
        return transition_mask

    def update(self, external_force: np.ndarray) -> bool:
        delta_state = self.compute_delta_state(external_force)
        self.prev_state = self.state.copy()
        self.state += delta_state
        self.goal = self.trigger_goal_transition()
        return self.stop_counter < self.stop_threshold

    def _get_valid_input(self, prompt: str, cast_type: t.Type, condition: t.Callable[[t.Any], bool]) -> t.Any:
        while True:
            try:
                user_input = cast_type(input(prompt))
                if condition(user_input):
                    return user_input
                else:
                    print("Invalid value. Please try again.")
            except ValueError:
                print("Invalid input format. Please enter a valid number.")

def main():
    model = BioHoloneticModel(n_dimensions=3, n_clusters=3, noise_sigma=0.1)
    
    for t in range(100):
        external_force = np.random.uniform(-0.5, 0.5, 3)
        continue_simulation = model.update(external_force)
        print(f"Time {t}: State = {model.state}, Goal = {model.goal}")
        
        if not continue_simulation:
            print("Simulation stopped due to delta exceeding threshold for consecutive iterations.")
            break

if __name__ == "__main__":
    main()
