import numpy as np
import typing as t
import matplotlib.pyplot as plt

class RegulationModel1:
    def __init__(self, n_dimensions: int, n_components: int, noise_sigma: float = 0.1):
        self.optimum_value_history = []  # Store the history of optimum values over time
        self.optimum_value_change_times = []  # Store the time steps when optimum values change
        self.n_dimensions = n_dimensions
        self.n_components = n_components
        self.noise_sigma = noise_sigma
        self.disruption_times = []  # Record times when disruption occurs
        
        self.optimum_values = np.array([
            [1, 2, 3],
            [3, 1, 2],
            [1.5, 1.5, 1.5]
        ])

        self.state = self.optimum_values[0].copy()  # Start with the first optimum value
        self.prev_state = self.optimum_values[0].copy()  # Start with the first optimum value

        self.components_positive = np.array([
            [0.1, 0.9, 0.5],
            [0.5, 0.1, 0.9],
            [0.9, 0.5, 0.1]
        ])

        print("Assigned Positive Components:")
        print(self.components_positive)
        
        self.cue_generator = np.random.RandomState()
        self.optimum_value = self.optimum_values[0]  # Start with the first optimum value
        self.stop_counter = 0
        self.stop_threshold = 5  # Updated stop threshold to 5 turns
        self.delta_history = []  # For storing raw delta values over time

    def compute_delta_state(self, external_force: np.ndarray) -> np.ndarray:
        gamma = np.array([1, 1, 1])
        delta = self.optimum_value + gamma - self.state
        self.delta_history.append(delta.copy())  # Store the raw delta vector
        print(f"Delta: {delta}")

        if np.any(np.abs(delta) > 1.5):
            self.stop_counter += 1
            print(f"Stop Counter Incremented to: {self.stop_counter}")  # Debug statement
        else:
            self.stop_counter = 0
            print(f"Stop Counter Reset to: {self.stop_counter}")  # Debug statement

        component_contributions = np.zeros(self.n_dimensions)
        alpha = 1 / (1 + np.exp(-1.5))
        beta = 1 / (1 + np.exp(-1.5))

        for i in range(self.n_dimensions):
            if np.abs(delta[i]) > 1.5:
                selected_component = self.components_positive[:, i]
                if delta[i] > 0:
                    regulated_component = alpha * selected_component
                else:
                    regulated_component = -beta * selected_component

                print(f"Dimension {i}: Regulated Component = {regulated_component}, Total Contribution = {component_contributions[i]}")
            else:
                print("No regulation is applied.")
                regulated_component = np.zeros_like(self.components_positive[:, i])

            component_contributions[i] += np.sum(regulated_component)

        noise = np.random.normal(0, self.noise_sigma, self.n_dimensions)
        external_force_with_noise = external_force + noise

        # Add random disruption to delta_state
        if np.random.randint(1, 11) == 1:  # 1 in 10 chance of disruption
            print("Random disruption activated!")
            delta_state = component_contributions + external_force_with_noise + np.array([-1, -1, -1])
            self.disruption_times.append(len(self.delta_history))  # Record the disruption time
        else:
            delta_state = component_contributions + external_force_with_noise

        return delta_state

    def trigger_optimum_value_transition(self, external_input: t.Optional[np.ndarray] = None) -> np.ndarray:
        if self.stop_counter >= 3:
            print("Out-of-range trigger activated. Switching to the third optimum value.")
            return self.optimum_values[2]

        cue = self.cue_generator.randint(1, 6)
        if cue == 5:
            if np.array_equal(self.optimum_value, self.optimum_values[1]):
                print("Cue-based trigger activated, but the second optimum value is already in use. Retaining current optimum value.")
                return self.optimum_value
            else:
                print("Cue-based trigger activated. Switching to the second optimum value.")
                return self.optimum_values[1]

        transition_mask = self._identify_transition_regions()
        if np.any(transition_mask):
            if np.array_equal(self.optimum_value, self.optimum_values[0]):
                print("State-based trigger activated, but the first optimum value is already in use. Retaining current optimum value.")
                return self.optimum_value
            else:
                print("State-based trigger activated. Switching to the first optimum value.")
                return self.optimum_values[0]

        print("No trigger activated. Retaining current optimum value.")
        return self.optimum_value

    def _identify_transition_regions(self) -> np.ndarray:
        transition_mask = (
            (2 <= self.state) & (self.state <= 2.5) |
            (-2.5 <= self.state) & (self.state <= -2)
        )
        return transition_mask
        

    def update(self, external_force: np.ndarray) -> bool:
        delta_state = self.compute_delta_state(external_force)
        self.prev_state = self.state.copy()
        self.state += delta_state
        new_optimum_value = self.trigger_optimum_value_transition()

        if not np.array_equal(self.optimum_value, new_optimum_value):
            self.optimum_value_change_times.append(len(self.optimum_value_history))  # Record the time step of the change
            print(f"Optimum value changed to: {new_optimum_value} at time {len(self.optimum_value_history)}")

        self.optimum_value = new_optimum_value
        self.optimum_value_history.append(self.optimum_value.copy())  # Store the optimum value history

        return self.stop_counter < self.stop_threshold

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

        # Annotate optimum value changes with soft grey
        soft_grey = '#B0B0B0'  # Hex code for a soft grey color
        for change_time in self.optimum_value_change_times:
            plt.axvline(x=change_time, color=soft_grey, linestyle='--', label=f'Optimum Value Change at {change_time}')
            plt.text(change_time, delta_array[change_time, 0], f'Optimum Value: {self.optimum_value_history[change_time]}', 
                     verticalalignment='bottom', horizontalalignment='right', rotation=90, color=soft_grey)

        # Add vertical lines to indicate disruption events
        for disruption_time in self.disruption_times:
            plt.axvline(x=disruption_time, color='red', linestyle='-', label=f'Disruption at {disruption_time}')
            plt.text(disruption_time, delta_array[disruption_time, 0], 'Disruption', 
                     verticalalignment='bottom', horizontalalignment='right', color='red')

        # Add the components as a text matrix on the plot
        component_text = "\n".join([f"Component {i+1}: {component}" for i, component in enumerate(self.components_positive)])
        plt.figtext(0.8, 0.5, component_text, fontsize=9, verticalalignment='center', horizontalalignment='left')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('delta_vs_time_with_components_and_optimum_value_changes.png')
        plt.show()
        
def main():
    model = RegulationModel1(n_dimensions=3, n_components=3, noise_sigma=0.1)
    
    for t in range(100):
        external_force = np.random.uniform(-0.3, 0.3, 3)
        continue_simulation = model.update(external_force)
        print(f"Time {t}: State = {model.state}, Optimum Value = {model.optimum_value}")
        
        if not continue_simulation:
            print("Simulation stopped due to delta exceeding threshold for consecutive iterations.")
            break

    model.plot_delta_vs_time()  # Call the plotting function after the simulation

if __name__ == "__main__":
    main()
