from neurons.digital_neuron import DigitalNeuron

class NeuronGroupManager:
    def __init__(self, similarity_threshold=0.6):
        self.neurons = []
        self.reports = []
        self.similarity_threshold = similarity_threshold  # Для расширения логики

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def process_tokens(self, tokens):
        self.reports.clear()
        any_activated = False

        for neuron in self.neurons:
            # Проверяем каждый токен, активируется ли нейрон хотя бы один раз
            activated = False
            for token in tokens:
                if neuron.process(token):
                    activated = True
            if activated:
                any_activated = True
            report = neuron.generate_report(tokens, activated)
            self.reports.append(report)

        if not any_activated and tokens:
            # Создаем новый нейрон для первого токена
            new_neuron = DigitalNeuron(token_scope=[tokens[0]])
            self.add_neuron(new_neuron)
            activated = False
            for token in tokens:
                if new_neuron.process(token):
                    activated = True
            report = new_neuron.generate_report(tokens, activated)
            self.reports.append(report)

    def train_neurons(self, tokens):
        trained_tokens_count = 0
        for neuron in self.neurons:
            # Добавляем в token_scope новые токены
            new_tokens = [t for t in tokens if t not in neuron.token_scope]
            for t in new_tokens:
                neuron.token_scope.append(t)
                trained_tokens_count += 1
        return trained_tokens_count
