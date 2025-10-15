# ia_core.py
"""
IA simples em Python usando um modelo grátis (distilgpt2 via transformers).
- Lê configuração de interface em ia_app.cro (JSON ou formato simples - veja load_ui_config).
- Expõe uma API Flask: POST /api/generate { "prompt": "...", "max_tokens": 50 }
- Também disponibiliza generate_response(prompt) para uso interno.
- Ajuste MODEL_NAME para outro modelo gratuito se desejar (ex: "gpt2", "microsoft/DialoGPT-small").
"""

import os
import json
import threading
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify

# transformers imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --------------- Configurações ---------------
MODEL_NAME = os.getenv("IA_MODEL", "distilgpt2")  # modelo grátis e pequeno
DEVICE = 0 if (os.getenv("IA_USE_GPU", "0") == "1") else -1  # use GPU se IA_USE_GPU=1
DEFAULT_MAX_TOKENS = 60

IA_APP_CRO = "ia_app.cro"  # arquivo de interface que você mencionou

# --------------- Carregamento de UI/config ---------------
def load_ui_config(path: str = IA_APP_CRO) -> Dict[str, Any]:
    """
    Tenta carregar ia_app.cro. Suporta:
     - JSON puro (recomendado): {"title": "...", "fields": [...], ...}
     - Se o arquivo não existir, retorna uma configuração padrão.
    """
    if not os.path.isfile(path):
        return {
            "title": "IA App (config padrão)",
            "description": "Interface gerada automaticamente porque ia_app.cro não foi encontrada.",
            "inputs": [
                {"id": "prompt", "label": "Prompt", "type": "textarea", "placeholder": "Escreva aqui..."}
            ],
            "buttons": [
                {"id": "send", "label": "Enviar"}
            ]
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            # tentativa: JSON
            try:
                cfg = json.loads(raw)
                return cfg
            except json.JSONDecodeError:
                # fallback: tentar interpretar como key=value por linha
                cfg = {"title": "ia_app (format inferido)", "raw_lines": []}
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cfg["raw_lines"].append(line)
                return cfg
    except Exception as e:
        return {"title": "Erro ao ler ia_app.cro", "error": str(e)}

# --------------- Inicializar modelo ---------------
_model_pipeline = None
_model_lock = threading.Lock()

def init_model(model_name: str = MODEL_NAME, device: int = DEVICE):
    """
    Inicializa o pipeline de geração (thread-safe).
    Use device = -1 para CPU ou 0/1 etc. para GPU.
    """
    global _model_pipeline
    with _model_lock:
        if _model_pipeline is not None:
            return _model_pipeline
        print(f"[IA] Carregando tokenizer e modelo '{model_name}' (device={device})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,  # -1 CPU, 0..n GPU
        )
        _model_pipeline = gen
        print("[IA] Modelo carregado com sucesso.")
        return _model_pipeline

# --------------- Geração de texto ---------------
def generate_response(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.8, top_k: int = 50) -> str:
    """
    Gera texto a partir do prompt usando o pipeline carregado.
    Retorna a string completa gerada.
    """
    if not prompt:
        return ""
    gen = init_model()
    # transformers' pipeline text-generation uses "max_length" absolute; we compute target length.
    # Simples abordagem: pedir max_length = prompt_tokens + max_tokens. Aqui usamos na prática um valor seguro.
    outputs = gen(
        prompt,
        max_length=len(prompt.split()) + max_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=1,
    )
    # outputs é lista de dicts com 'generated_text'
    text = outputs[0].get("generated_text", "")
    # Em alguns casos o modelo repete o prompt; remover prompt no início se presente.
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()

# --------------- Flask API ---------------
def create_app() -> Flask:
    app = Flask(__name__)
    ui_cfg = load_ui_config()

    @app.route("/api/ui", methods=["GET"])
    def get_ui():
        return jsonify(ui_cfg)

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get("prompt") or data.get("text") or ""
        max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(data.get("temperature", 0.8))
        top_k = int(data.get("top_k", 50))
        if not prompt:
            return jsonify({"error": "prompt vazio"}), 400
        try:
            result = generate_response(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k)
            return jsonify({"prompt": prompt, "response": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/", methods=["GET"])
    def index():
        # Resposta simples; a UI real pode ler /api/ui e montar a interface.
        return (
            "<h2>IA App</h2>"
            "<p>Use <code>POST /api/generate</code> com JSON {\"prompt\": \"...\"}.</p>"
            "<p>Leia /api/ui para configuração da interface (ia_app.cro).</p>"
        )

    return app

# --------------- CLI rápido ---------------
def run_cli():
    print("== IA CLI == (digite 'exit' para sair)")
    print("Prompt examples: 'Olá, quem é você?'")
    init_model()  # carrega modelo antes do loop
    while True:
        try:
            prompt = input("\nPrompt> ")
        except (KeyboardInterrupt, EOFError):
            print("\nSaindo...")
            break
        if not prompt:
            continue
        if prompt.strip().lower() in ("exit", "quit"):
            print("Saindo...")
            break
        resp = generate_response(prompt, max_tokens=80)
        print("\nResposta:\n", resp)

# --------------- Entrada principal ---------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IA Core - servidor/CLI simples")
    parser.add_argument("--serve", action="store_true", help="Rodar servidor Flask (porta 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host para Flask")
    parser.add_argument("--port", default=5000, type=int, help="Porta para Flask")
    parser.add_argument("--cli", action="store_true", help="Rodar CLI interativo")
    parser.add_argument("--model", default=MODEL_NAME, help="Nome do modelo HuggingFace")
    args = parser.parse_args()

    # sobrescreve o MODEL_NAME e força recarregar se necessário
    MODEL = args.model
    if args.serve:
        # inicializa modelo em thread principal para reduzir latência no primeiro pedido
        init_model(model_name=MODEL, device=DEVICE)
        app = create_app()
        print(f"Rodando servidor em http://{args.host}:{args.port} (UI file: {IA_APP_CRO})")
        app.run(host=args.host, port=args.port)
    elif args.cli:
        init_model(model_name=MODEL, device=DEVICE)
        run_cli()
    else:
        print("Nenhuma ação especificada. Use --serve para servidor ou --cli para linha de comando.")
        print("Exemplo: python ia_core.py --serve")
