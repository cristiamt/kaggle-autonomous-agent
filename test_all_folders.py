import subprocess
import time

print("Iniciando Bateria de Testes: Kaggle Autonomous Agent")
print("-" * 50)

for i in range(1, 17):
    problem_name = f"kaggle_in_kaggle_train_{i:02d}"

    print(f"\nDisparando Agente para: {problem_name}")

    comando = [
        "python", "run_local_eval.py",
        "--problem", problem_name
    ]

    try:
        subprocess.run(comando, check=True)
        print(f"[OK] Concluido: {problem_name}")

    except subprocess.CalledProcessError:
        print(f"[ERRO] Falha ou erro na execucao do {problem_name}.")

    if i < 16:
        print("Pausando por 30 segundos para evitar bloqueio da API do Gemini (RPM)...")
        time.sleep(30)

print("\nBateria de testes finalizada! Verifique os relatorios na pasta output/.")
