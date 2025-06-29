import time
import uuid
import difflib
import math

class DigitalNeuron:
    def __init__(self, token_scope=None, context_window=5):
        self.id = str(uuid.uuid4())
        self.token_scope = token_scope or []
        self.activation_score = 0.0
        self.context_window = context_window  # сколько токенов слева/справа учитываем
        self.activation_threshold = 1.0       # порог активации нейрона

    def token_similarity(self, token1, token2):
        # Коэффициент схожести строк (можно заменить на векторное сходство)
        return difflib.SequenceMatcher(None, token1, token2).ratio()

    def positional_weight(self, token_index, matched_index):
        # Экспоненциальное затухание веса по расстоянию между токенами
        distance = abs(token_index - matched_index)
        if distance > self.context_window:
            return 0.0
        return math.exp(-distance)

    def process_with_attention(self, input_tokens):
        """
        Обрабатывает список токенов с учётом внимания и позиций.
        Возвращает True, если итоговый скор >= порога активации.
        """
        attention_scores = [0.0] * len(input_tokens)

        for neuron_token in self.token_scope:
            best_score = 0.0
            best_pos = -1
            for i, input_token in enumerate(input_tokens):
                sim = self.token_similarity(input_token, neuron_token)
                if sim > best_score:
                    best_score = sim
                    best_pos = i

            if best_pos >= 0 and best_score > 0:
                start = max(0, best_pos - self.context_window)
                end = min(len(input_tokens), best_pos + self.context_window + 1)
                for i in range(start, end):
                    pw = self.positional_weight(i, best_pos)
                    attention_scores[i] += best_score * pw

        self.activation_score = sum(attention_scores) / max(1, len(input_tokens))
        return self.activation_score >= self.activation_threshold

    def generate_report(self, activated):
        return {
            'neuron_id': self.id,
            'timestamp': time.time(),
            'activated': activated,
            'activation_score': self.activation_score,
            'token_scope': self.token_scope,
        }

    def train(self, input_tokens, expansion_threshold=0.8):
        """
        Расширяет token_scope новыми токенами, похожими на существующие,
        если их сходство превышает expansion_threshold.
        """
        new_tokens = set()
        for token in input_tokens:
            if token not in self.token_scope:
                for scope_token in self.token_scope:
                    if self.token_similarity(token, scope_token) >= expansion_threshold:
                        new_tokens.add(token)
                        break
        if new_tokens:
            self.token_scope.extend(new_tokens)
            self.token_scope = list(set(self.token_scope))  # убираем дубликаты
        return len(new_tokens)
