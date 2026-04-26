import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 1. API КЛЮЧ ======
import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ====== 2. URL API ======
MODELS_URL = "https://openrouter.ai/api/v1/models"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# ====== 3. Заголовки ======
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY не задан. Установи переменную окружения.")

# ====== 4. Получаем список моделей ======
def get_free_models():
    response = requests.get(MODELS_URL, headers=headers)
    
    if response.status_code != 200:
        print("Ошибка при получении моделей:", response.text)
        return []

    data = response.json()
    free_models = []

    for model in data.get("data", []):
        pricing = model.get("pricing", {})
        
        if pricing.get("prompt") == "0":
            model_id = model.get("id")

            # Пытаемся вытащить параметры типа 7B, 13B и т.д.
            match = re.search(r"(\d+)([Bb])", model_id)
            params = match.group(1) + match.group(2) if match else "N/A"

            free_models.append({
                "id": model_id,
                "params": params
            })

    return free_models

# ====== 5. Проверка модели ======
def test_model(model):
    model_id = model["id"]

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Say 'ok' in one word"}
        ],
        "max_tokens": 5
    }

    start = time.time()

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=10)
        elapsed = round(time.time() - start, 2)

        if response.status_code == 200:
            return ("OK", elapsed)
        else:
            return (f"ERR {response.status_code}", elapsed)

    except requests.exceptions.Timeout:
        return ("TIMEOUT", None)
    except Exception:
        return ("FAIL", None)

# ====== 6. Основной запуск ======
def main():
    print("Получаем список бесплатных моделей...\n")
    
    models = get_free_models()

    if not models:
        print("Нет доступных моделей.")
        return

    print(f"Найдено моделей: {len(models)}\n")

    # Dynamically calculate max model name length
    max_len = max(len(m['id']) for m in models) if models else 10
    col_width = max(max_len + 2, 20)

    print(f"{'Model':{col_width}} | {'Params':8} | {'Status':10} | {'Time(s)':8}")
    print("-" * (col_width + 32))

    fastest = None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(test_model, m): m for m in models}

        for future in as_completed(futures):
            model = futures[future]
            status, elapsed = future.result()

            name = model["id"]
            params = model["params"]

            time_str = str(elapsed) if elapsed is not None else "-"

            print(f"{name:{col_width}} | {params:8} | {status:10} | {time_str:8}")

            if status == "OK" and elapsed is not None:
                if fastest is None or elapsed < fastest[2]:
                    fastest = (name, params, elapsed)

    if fastest:
        print("\n Самая быстрая модель:")
        name, params, elapsed = fastest
        time_str = str(elapsed)
        print("-" * (col_width + 32))
        print(f"{name:{col_width}} | {params:8} | {'OK':10} | {time_str:8}")
        print("-" * (col_width + 32))
    else:
        print("\nНет рабочих моделей.")

# ====== 7. Точка входа ======
if __name__ == "__main__":
    main()