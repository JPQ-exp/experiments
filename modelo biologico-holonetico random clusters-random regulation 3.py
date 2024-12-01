import numpy as np
import typing as t
import matplotlib.pyplot as plt

class BioHoloneticModel:
    def __init__(self, n_dimensions: int, n_clusters: int, noise_sigma: float = 0.1):
        self.goal_history = []  # Store the history of goals over time
        self.goal_change_times = []  # Store the time steps when goals change
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

        # Print the generated clusters
        print("Generated Positive Clusters:")
        print(self.clusters_positive)
        
        self.cue_generator = np.random.RandomState()
        self.goal = self.goals[0]  # Start with the first goal
        self.stop_counter = 0

        self.stop_threshold = self._get_valid_input(
            "Please enter the stop threshold (a positive integer): ", 
            int, 
            lambda x: x > 0
        )

        self.delta_history = []  # For storing raw delta values over time
        self.regulation_factor = 1.5  # Set the regulation factor to always be 1.5

    def _input_clusters(self) -> np.ndarray:
        clusters = []
        for _ in range(self.n_clusters):
            cluster_values = np.round(np.random.uniform(0.1, 0.9, self.n_dimensions), 1)
            clusters.append(cluster_values)
        return np.array(clusters)

    def compute_delta_state(self, external_force: np.ndarray) -> np.ndarray:
        delta = self.goal - self.state
        self.delta_history.append(delta.copy())  # Store the raw delta vector
        print(f"Delta: {delta}")

        if np.any(np.abs(delta) > 2):
            self.stop_counter += 1
            print(f"Stop Counter Incremented to: {self.stop_counter}")  # Debug statement
        else:
            self.stop_counter = 0
            print(f"Stop Counter Reset to: {self.stop_counter}")  # Debug statement

        cluster_contributions = np.zeros(self.n_dimensions)

        for i in range(self.n_dimensions):
            regulation_factor = 1  # Default to no regulation
            if np.abs(delta[i]) > 2:  # Regulate when out of range
                regulation_factor = self._compute_dynamic_regulation(delta, i)
                print(f"Regulation factor for dimension {i}: {regulation_factor}")

            # Randomly choose between positive and negative clusters
            is_positive = np.random.choice([True, False])
            selected_cluster_index = np.random.randint(0, self.n_clusters)
            if is_positive:
                regulated_cluster = self.clusters_positive[selected_cluster_index] * regulation_factor
                print(f"Regulated positive cluster for dimension {i}: {regulated_cluster}")
            else:
                regulated_cluster = self.clusters_negative[selected_cluster_index] * regulation_factor
                print(f"Regulated negative cluster for dimension {i}: {regulated_cluster}")

            # Add the regulated contribution of the selected cluster
            cluster_contributions[i] += regulated_cluster[i]

                # Add contributions from all other clusters, avoiding repetition
            for j in range(self.n_clusters):
                if j != selected_cluster_index:
                    cluster_contributions[i] += self.clusters_positive[j, i] 
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
        if self.stop_counter >= 3:
            print("Out-of-range trigger activated. Switching to the third goal.")
            return self.goals[2]

        cue = self.cue_generator.randint(1, 6)
        if cue == 5:
            print("Cue-based trigger activated. Switching to the second goal.")
            return self.goals[1]

        transition_mask = self._identify_transition_regions()
        if np.any(transition_mask):
            print("State-based trigger activated. Switching to the first goal.")
            return self.goals[0]

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
        new_goal = self.trigger_goal_transition()

        if not np.array_equal(self.goal, new_goal):
            self.goal_change_times.append(len(self.goal_history))  # Record the time step of the change
            print(f"Goal changed to: {new_goal} at time {len(self.goal_history)}")

        self.goal = new_goal
        self.goal_history.append(self.goal.copy())  # Store the goal history

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

    def plot_delta_vs_time(self):
        plt.figure(figsize=(12, 6))

        # Plot delta values for each dimension over time
        delta_array = np.array(self.delta_history)
        for dim in range(self.n_dimensions):
            plt.plot(delta_array[:, dim], label=f'Delta Dimension {dim+1}')

        plt.title('Delta Values vs Time')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Delta Values')
        plt.legend()
        plt.grid(True)

        # Annotate goal changes with soft grey
        soft_grey = '#B0B0B0'  # Hex code for a soft grey color
        for change_time in self.goal_change_times:
            plt.axvline(x=change_time, color=soft_grey, linestyle='--', label=f'Goal Change at {change_time}')
            plt.text(change_time, delta_array[change_time, 0], f'Goal: {self.goal_history[change_time]}', 
                    verticalalignment='bottom', horizontalalignment='right', rotation=90, color=soft_grey)
        
        # Add the clusters as a text matrix on the plot
        cluster_text = "\n".join([f"Cluster {i+1}: {cluster}" for i, cluster in enumerate(self.clusters_positive)])
        plt.figtext(0.8, 0.5, cluster_text, fontsize=9, verticalalignment='center', horizontalalignment='left')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('delta_vs_time_with_clusters_and_goal_changes.png')
        plt.show()

def main():
    model = BioHoloneticModel(n_dimensions=3, n_clusters=3, noise_sigma=0.1)
    
    for t in range(100):
        external_force = np.random.uniform(-0.5, 0.5, 3)
        continue_simulation = model.update(external_force)
        print(f"Time {t}: State = {model.state}, Goal = {model.goal}")
        
        if not continue_simulation:
            print("Simulation stopped due to delta exceeding threshold for consecutive iterations.")
            break

    model.plot_delta_vs_time()  # Call the plotting function after the simulation

if __name__ == "__main__":
    main()
