import sys
import os
import tkinter as tk
from tkinter import scrolledtext
from collections import defaultdict
import uuid
import time
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))

class DigitalNeuron:
    def __init__(self, token_scope=None, max_scope=30):
        self.id = str(uuid.uuid4())
        self.token_scope = token_scope or []
        self.activation_count = 0
        self.max_scope = max_scope

    def process(self, token):
        activated = token in self.token_scope
        if activated:
            self.activation_count += 1
        return activated

    def generate_report(self, activated):
        return {
            'neuron_id': self.id,
            'timestamp': time.time(),
            'activated': activated,
            'activation_count': self.activation_count,
            'token_scope': list(self.token_scope),
        }

    def train(self, tokens):
        added = 0
        for t in tokens:
            if t not in self.token_scope:
                if len(self.token_scope) < self.max_scope:
                    self.token_scope.append(t)
                    added += 1
        return added

class RussianTokenizer:
    def tokenize(self, text):
        return [w.strip('.,!?;:()[]').lower() for w in text.strip().split() if w.strip('.,!?;:()[]')]

class NeuronGroupManager:
    def __init__(self):
        self.neurons = []
        self.reports = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def process_tokens(self, tokens):
        self.reports.clear()
        any_activated = False
        activated_neurons = set()

        for token in tokens:
            activated_for_token = False
            for neuron in self.neurons:
                activated = neuron.process(token)
                if activated:
                    any_activated = True
                    activated_for_token = True
                    activated_neurons.add(neuron)
                report = neuron.generate_report(activated)
                self.reports.append(report)

            if not activated_for_token:
                new_neuron = DigitalNeuron(token_scope=[token])
                self.add_neuron(new_neuron)
                new_neuron.process(token)
                report = new_neuron.generate_report(True)
                self.reports.append(report)
                any_activated = True
                activated_neurons.add(new_neuron)

    def get_reports(self):
        return self.reports

    def train_neurons(self, tokens):
        added_total = 0
        for neuron in self.neurons:
            added = neuron.train(tokens)
            added_total += added
        return added_total

    def cluster_neurons(self, similarity_threshold=0.3):
        clustered = []
        used = set()

        for i, neuron_a in enumerate(self.neurons):
            if i in used:
                continue
            cluster = [neuron_a]
            tokens_a = set(neuron_a.token_scope)

            for j, neuron_b in enumerate(self.neurons):
                if i != j and j not in used:
                    tokens_b = set(neuron_b.token_scope)
                    intersection = tokens_a & tokens_b
                    union = tokens_a | tokens_b
                    similarity = len(intersection) / len(union) if union else 0

                    if similarity > similarity_threshold:
                        cluster.append(neuron_b)
                        used.add(j)

            if len(cluster) > 1:
                merged_tokens = list(set().union(*(n.token_scope for n in cluster)))
                cluster[0].token_scope = merged_tokens[:cluster[0].max_scope]
                for n in cluster[1:]:
                    self.neurons.remove(n)

            clustered.append(cluster[0])
            used.add(i)

        self.neurons = clustered

class UnderMind:
    def __init__(self, reward_threshold=1):
        self.reward_threshold = reward_threshold
        self.neuron_rewards = defaultdict(int)

    def receive_reports(self, reports):
        for r in reports:
            if r['activated']:
                self.neuron_rewards[r['neuron_id']] += 1

    def evaluate_rewards(self):
        rewarded = []
        for neuron_id, count in self.neuron_rewards.items():
            if count >= self.reward_threshold:
                rewarded.append(neuron_id)
        self.neuron_rewards.clear()
        return rewarded

class CeaAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CEA-AI Interface")

        self.text_input = tk.Text(root, height=5, width=60)
        self.text_input.pack(pady=5)
        self.text_input.focus_set()

        self.paste_button = tk.Button(root, text="📋 Вставить из буфера", command=self.paste_text_input)
        self.paste_button.pack(pady=2)

        self.clear_button = tk.Button(root, text="🗑 Очистить ввод", command=self.clear_text_input)
        self.clear_button.pack(pady=2)

        self.process_button = tk.Button(root, text="Обработать текст", command=self.process_text)
        self.process_button.pack(pady=2)

        self.cluster_button = tk.Button(root, text="🔀 Кластеризовать нейроны", command=self.cluster_neurons)
        self.cluster_button.pack(pady=2)

        self.restart_button = tk.Button(root, text="🔁 Перезапустить модель", command=self.restart_model)
        self.restart_button.pack(pady=2)

        self.copy_log_button = tk.Button(root, text="📋 Копировать лог", command=self.copy_log)
        self.copy_log_button.pack(pady=2)

        self.log_output = scrolledtext.ScrolledText(root, height=20, width=80, state='disabled')
        self.log_output.pack(pady=5)

        self.text_input.bind('<Control-v>', self.paste_text_input)
        self.text_input.bind('<Control-V>', self.paste_text_input)
        self.text_input.bind('<Control-c>', self.copy_text_input)
        self.text_input.bind('<Control-C>', self.copy_text_input)
        self.text_input.bind('<Control-l>', self.clear_text_input)
        self.text_input.bind('<Control-L>', self.clear_text_input)

        self.log_output.bind('<Control-c>', self.copy_log)
        self.log_output.bind('<Control-C>', self.copy_log)

        self.tokenizer = RussianTokenizer()
        self.init_model()

    def init_model(self):
        self.neuron_manager = NeuronGroupManager()
        self.under_mind = UnderMind(reward_threshold=1)
        self.neuron_manager.add_neuron(DigitalNeuron(token_scope=["почему", "как"]))
        self.neuron_manager.add_neuron(DigitalNeuron(token_scope=["счастливый", "грустный"]))
        self.neuron_manager.add_neuron(DigitalNeuron(token_scope=["бежать", "прыгать", "ходить"]))
        self.log("✅ Модель инициализирована с начальными нейронами.")

    def log(self, message):
        self.log_output.config(state='normal')
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.see(tk.END)
        self.log_output.config(state='disabled')

    def process_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            self.log("❗ Введите текст для обработки.")
            return
        tokens = self.tokenizer.tokenize(text)
        self.log(f"📥 Токены: {tokens}")

        self.neuron_manager.process_tokens(tokens)
        reports = self.neuron_manager.get_reports()

        added = self.neuron_manager.train_neurons(tokens)
        self.log(f"🎓 Добавлено новых токенов к областям нейронов: {added}")

        activity_map = defaultdict(int)
        for r in reports:
            if r['activated']:
                activity_map[r['neuron_id']] += 1

        self.log("📊 Активность нейронов:")
        for neuron_id, count in activity_map.items():
            self.log(f" - Нейрон {neuron_id[:8]}... активность: {count}")

        self.under_mind.receive_reports(reports)
        rewarded = self.under_mind.evaluate_rewards()
        self.log("🏆 Награждённые нейроны:")
        if rewarded:
            for neuron_id in rewarded:
                self.log(f" - ✅ {neuron_id[:8]}...")
        else:
            self.log(" - Нет нейронов, превысивших порог.")

        self.log("—" * 40)

    def cluster_neurons(self):
        self.neuron_manager.cluster_neurons()
        self.log("🔀 Кластеризация нейронов завершена.")

    def restart_model(self):
        self.log("🔁 Перезапуск модели...")
        self.init_model()
        self.log_output.config(state='normal')
        self.log_output.delete(1.0, tk.END)
        self.log_output.config(state='disabled')
        self.log("✅ Модель очищена и перезапущена.")

    def clear_text_input(self, event=None):
        self.text_input.delete("1.0", tk.END)
        return "break"

    def copy_log(self, event=None):
        self.root.clipboard_clear()
        log_text = self.log_output.get("1.0", tk.END)
        self.root.clipboard_append(log_text)
        self.log("📋 Лог скопирован в буфер обмена.")
        return "break"

    def copy_text_input(self, event=None):
        try:
            selected = self.text_input.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
        except tk.TclError:
            pass
        return "break"

    def paste_text_input(self, event=None):
        try:
            text = self.root.clipboard_get()
            self.text_input.insert(tk.INSERT, text)
        except tk.TclError:
            pass
        return "break"

if __name__ == "__main__":
    root = tk.Tk()
    app = CeaAIApp(root)
    root.mainloop()
