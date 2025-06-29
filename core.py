class UnderMind:
    def __init__(self, reward_threshold=1):
        self.reward_threshold = reward_threshold
        self.reports = []

    def reset(self):
        self.reports = []

    def receive_reports(self, reports):
        self.reports = reports

    def evaluate_rewards(self):
        rewarded_neurons = []
        for report in self.reports:
            if report['activation_count'] >= self.reward_threshold:
                rewarded_neurons.append(report['neuron_id'])
        return rewarded_neurons
