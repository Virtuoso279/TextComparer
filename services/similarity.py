import re
import math
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash


class StylisticFingerprint:
    """
    Клас для створення та порівняння стилістичних "відбитків" тексту.
    """
    def __init__(self, text: str):
        self.text = text
        self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> list[float]:
        # Розділяємо текст на речення, використовуючи розділові знаки ., !, ?
        # Видаляємо порожні рядки після розбиття
        sentences = [s.strip() for s in re.split(r'[.!?]', self.text) if s.strip()]

        # Витягаємо всі слова (послідовність букв/цифр) у нижньому регістрі
        words = re.findall(r"\b\w+\b", self.text.lower())

        # Підраховуємо всі знаки пунктуації (кома, крапка, знак оклику тощо)
        punctuation = re.findall(r'[,.!?;:]', self.text)

        # Обчислюємо середню довжину речення у словах - Якщо речень немає, повертаємо 0.0
        avg_sentence = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0.0
        # Обчислюємо середню довжину слова у символах - Якщо слів немає, повертаємо 0.0
        avg_word = sum(len(w) for w in words) / len(words) if words else 0.0
        # Обчислюємо щільність пунктуації як кількість знаків пунктуації на слово
        punc_density = len(punctuation) / len(words) if words else 0.0
        # Лексичне різноманіття = кількість унікальних слів / загальна кількість слів
        lex_div = len(set(words)) / len(words) if words else 0.0
        # Частка коротких речень (менше 5 слів)
        short_ratio = len([s for s in sentences if len(s.split()) < 5]) / len(sentences) if sentences else 0.0

        return [round(x, 3) for x in (avg_sentence, avg_word, punc_density, lex_div, short_ratio)]

    @staticmethod
    def compare(fp1: list[float], fp2: list[float], epsilon: float = 1e-8) -> float:
        """
       Порівнює два стилістичні вектори fp1 та fp2.
       Використовує нормалізовану Манхеттенську відстань:
       для кожної компоненти обчислюється abs(a - b) / (max(a, b) + epsilon),
       після чого всі нормалізовані різниці усереднюються, і від 1 віднімається
       отримане середнє, щоб отримати значення у [0, 1].
        """
        diffs = []
        for a, b in zip(fp1, fp2):
            if a == 0 and b == 0:
                diffs.append(0.0)
            else:
                diffs.append(abs(a - b) / (max(a, b) + epsilon))
        score = 1 - (sum(diffs) / len(diffs)) if diffs else 0.0
        return round(score, 4)


class JaccardSimilarity:
    """
    Клас для обчислення Jaccard-схожості між двома текстами.
    """
    @staticmethod
    def compute(text1: str, text2: str) -> float:
        # Створюємо множину унікальних слів з першого тексту
        set1 = set(re.findall(r"\b\w+\b", text1.lower()))
        # Створюємо множину унікальних слів з другого тексту
        set2 = set(re.findall(r"\b\w+\b", text2.lower()))

        # Обчислюємо об’єднання та перетин множин
        union = set1 | set2
        inter = set1 & set2

        # Якщо об’єднання порожнє (обидва тексти не містять слів), повертаємо 0.0
        # Інакше повертаємо відношення потужності перетину до потужності об’єднання
        return round(len(inter) / len(union), 4) if union else 0.0


class TfIdfSimilarity:
    """
    Клас для обчислення TF-IDF векторів і косинусної схожості.
    """
    def __init__(self, texts: list[str]):
        # Зберігаємо список текстів (зазвичай два текстові документи)
        self.texts = texts
        # Побудова спільного словника (vocab) із унікальних слів усіх текстів
        self.vocab = self._build_vocab()
        # Обчислення зворотної частоти документів (IDF) для кожного слова словника
        self.idf = self._compute_idf()

    def _build_vocab(self) -> list[str]:
        # Токенізуємо кожен текст (виділяємо слова у нижньому регістрі)
        tokenized = [re.findall(r"\b\w+\b", t.lower()) for t in self.texts]
        # Створюємо множину усіх унікальних слів і повертаємо відсортований список
        return sorted({w for toks in tokenized for w in toks})

    def _compute_idf(self) -> dict[str, float]:
        N = len(self.texts)
        # Знову токенізуємо всі тексти
        tokenized = [re.findall(r"\b\w+\b", t.lower()) for t in self.texts]
        # Для кожного слова словника рахуємо, у скількох документах воно зустрічається (df)
        df = {w: sum(1 for toks in tokenized if w in toks) for w in self.vocab}
        # Обчислюємо згладжену IDF: log((N+1)/(df[w]+1)) + 1
        return {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in self.vocab}

    def compute_vectors(self) -> list[list[float]]:
        """
        Повертає список TF-IDF векторів для кожного тексту.
        Кроки:
        1) Токенізуємо текст.
        2) Для кожного слова словника рахуємо tf = 1 + log(count), якщо count>0, інакше 0.
        3) Множимо tf на відповідну idf[w] для формування елементу вектора.
        4) Застосовуємо L2-нормалізацію: ділити кожен елемент вектора на його довжину.
        """
        vectors = []
        for text in self.texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            vec = []
            for w in self.vocab:
                count = tokens.count(w)
                tf = 1 + math.log(count) if count > 0 else 0.0
                vec.append(tf * self.idf[w])
            norm = math.sqrt(sum(x * x for x in vec))
            vectors.append([x / norm for x in vec] if norm else vec)
        return vectors

    @staticmethod
    def compare(vec1: list[float], vec2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return round(dot, 4) if vec1 and vec2 else 0.0


class TextComparer:
    """
    Головний клас для порівняння двох текстів за всіма метриками.
    """
    def __init__(self, text1: str, text2: str):
        self.text1 = text1
        self.text2 = text2

    def compare_all(self) -> dict[str, float]:
        sf1 = StylisticFingerprint(self.text1).fingerprint
        sf2 = StylisticFingerprint(self.text2).fingerprint
        style_score = StylisticFingerprint.compare(sf1, sf2)
        jaccard_score = JaccardSimilarity.compute(self.text1, self.text2)
        tfidf = TfIdfSimilarity([self.text1, self.text2])
        v1, v2 = tfidf.compute_vectors()
        tfidf_score = TfIdfSimilarity.compare(v1, v2)
        return {
            'stylistic': style_score,
            'jaccard': jaccard_score,
            'tfidf': tfidf_score
        }


class User:
    """Простий POJO-клас користувача з гешованим паролем."""
    def __init__(self, username: str, password_hash: str):
        self.username = username
        self.password_hash = password_hash


class AuthService:
    """Сервіс для реєстрації та аутентифікації користувачів з flat-file JSON."""
    def __init__(self, file_path: str = 'users.json'):
        self.file_path = file_path
        self._users: dict[str, User] = {}
        self._load_users()

    def _load_users(self) -> None:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for username, pw_hash in data.items():
                self._users[username] = User(username, pw_hash)
        else:
            # створити порожній файл
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _save_users(self) -> None:
        data = {u: self._users[u].password_hash for u in self._users}
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def register(self, username: str, password: str) -> bool:
        if username in self._users:
            return False
        pw_hash = generate_password_hash(password)
        self._users[username] = User(username, pw_hash)
        self._save_users()
        return True

    def authenticate(self, username: str, password: str) -> bool:
        user = self._users.get(username)
        if not user:
            return False
        return check_password_hash(user.password_hash, password)
