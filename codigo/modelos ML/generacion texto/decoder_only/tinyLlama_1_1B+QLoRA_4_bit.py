import os
import torch
import pandas as pd
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer, SFTConfig


# CONFIGURACIÓN
class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    OUTPUT_DIR = os.path.join(MODEL_DIR, 'tinyllama_qlora')
    ADAPTER_PATH = os.path.join(OUTPUT_DIR, 'adapter')
    
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Configuración de cuantización (4-bit)
    LOAD_IN_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT = True
    
    # Configuración LoRA
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    LORA_BIAS = "none"
    LORA_TASK_TYPE = "CAUSAL_LM"
    
    # Entrenamiento
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    EPOCHS = 3
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 512
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.01
    
    # Generación
    MAX_NEW_TOKENS = 150
    MIN_NEW_TOKENS = 15
    TEMPERATURE = 0.8
    TOP_P = 0.95
    TOP_K = 50
    REPETITION_PENALTY = 1.05
    
    # Dispositivo
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# PREPARACIÓN DE DATOS

def load_miguel_data():
    df = pd.read_csv(Config.DATASET_PATH)
    miguel_df = df[df['speaker'] == 'MIGUEL'].copy()
    
    return miguel_df['text'].tolist()


def create_training_dataset(texts):
    
    formatted_data = []
    
    for i, text in enumerate(texts):
        words = text.split()
        if len(words) < 5:
            continue
            
        split_point = min(len(words) // 3, 10)
        start = ' '.join(words[:split_point])
        continuation = ' '.join(words[split_point:])
        
        formatted_text = f"""<|user|>
{start}</s>
<|assistant|>
{continuation}</s>"""
        
        formatted_data.append({"text": formatted_text})
    
    return Dataset.from_list(formatted_data)


# MODELO

def load_quantized_model():
    
    # Configuración de cuantización
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=Config.LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=Config.BNB_4BIT_USE_DOUBLE_QUANT,
    )
    
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
    )
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Asegurar consistencia con el modelo
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def setup_lora(model):
    
    # Preparar modelo para entrenamiento k-bit
    model = prepare_model_for_kbit_training(model)
    
    # Configuración LoRA
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.LORA_TARGET_MODULES,
        bias=Config.LORA_BIAS,
        task_type=Config.LORA_TASK_TYPE,
    )
    
    # Aplicar LoRA
    model = get_peft_model(model, lora_config)
    
    # Mostrar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return model


# ENTRENAMIENTO

def train_model():
    
    print("FINE-TUNING TINYLLAMA 1.1B CON QLORA (4-bit)")
    
    # Verificar CUDA
    if not torch.cuda.is_available():
        print("  ADVERTENCIA: No se detectó GPU. El entrenamiento será muy lento.")
        print("  Se recomienda usar Google Colab o un servidor con GPU.")
    else:
        print(f"  GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Cargar datos
    texts = load_miguel_data()
    full_dataset = create_training_dataset(texts)
    
    # Split dataset
    dataset_dict = full_dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']
    
    # Cargar modelo y tokenizer
    model, tokenizer = load_quantized_model()
    
    # Configurar LoRA
    model = setup_lora(model)
    
    # Argumentos de entrenamiento
    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        max_length=Config.MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Entrenar
    print("INICIANDO ENTRENAMIENTO")
    
    trainer.train()
    
    # Guardar adapter LoRA
    model.save_pretrained(Config.ADAPTER_PATH)
    tokenizer.save_pretrained(Config.ADAPTER_PATH)
    
    print("ENTRENAMIENTO COMPLETADO")


# GENERACIÓN

def load_finetuned_model():
    
    # Configuración de cuantización
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Cargar modelo base
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
    )
    
    # Cargar adapter LoRA
    model = PeftModel.from_pretrained(model, Config.ADAPTER_PATH)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Asegurar consistencia con el modelo
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, 
                  top_p=0.9, top_k=50, repetition_penalty=1.1):
    formatted_prompt = f"""<|user|>
{prompt}</s>
<|assistant|>
"""
    
    # Tokenizar
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Configuración de generación
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        min_new_tokens=Config.MIN_NEW_TOKENS,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decodificar
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|assistant|>" in generated_text:
        response = generated_text.split("<|assistant|>")[-1]
        response = response.replace("</s>", "").strip()
    else:
        response = generated_text
        
    # Devolver el texto completo (prompt + continuación)
    return f"{prompt} {response}"


# DEMO INTERACTIVA

def demo():
    
    print("DEMO")
    print("(TinyLlama 1.1B + QLoRA)")
    
    # Verificar que existe el modelo
    if not os.path.exists(Config.ADAPTER_PATH):
        print("\n No se encontró el modelo fine-tuneado.")
        print(f"  Ruta esperada: {Config.ADAPTER_PATH}")
        print("  Ejecuta primero: python tinyLlama_1_1B+QLoRA_4_bit.py --train")
        return
    
    # Cargar modelo
    try:
        model, tokenizer = load_finetuned_model()
        print(" Modelo cargado correctamente")
    except Exception as e:
        print(f" Error al cargar el modelo: {e}")
        return
    
    print("\nEscribe el inicio de una frase para que el modelo la complete.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        user_input = input("Tu texto: ").strip()
        
        if user_input.lower() == 'salir':
            break
        
        if not user_input:
            print("Por favor, escribe algo.")
            continue
        
        try:
            print("\nGenerando...")
            response = generate_text(
                model, tokenizer, user_input,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                repetition_penalty=Config.REPETITION_PENALTY
            )
            
            print(f"\nGenerado: {response}\n")
            
        except Exception as e:
            print(f"Error durante la generación: {e}")


# INSTALACIÓN DE DEPENDENCIAS

def check_dependencies():
    
    required = [
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "trl>=0.7.0",
    ]
    
    missing = []
    for req in required:
        package = req.split(">=")[0]
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f" Paquetes faltantes: {', '.join(missing)}")
        print("  Instala con: pip install " + " ".join(missing))
        return False
    
    print(" Todas las dependencias están instaladas")
    return True


# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tuning de TinyLlama 1.1B con QLoRA - Estilo Miguel Quintana'
    )
    parser.add_argument('--train', action='store_true', 
                        help='Entrenar el modelo')
    parser.add_argument('--demo', action='store_true', 
                        help='Ejecutar demo interactiva')
    parser.add_argument('--generate', type=str, 
                        help='Generar texto completando el prompt')
    parser.add_argument('--topic', type=str, 
                        help='Generar texto libre sobre un tema')
    parser.add_argument('--check', action='store_true', 
                        help='Verificar dependencias')
    parser.add_argument('--temperature', type=float, default=0.7, 
                        help='Temperatura para generación')
    parser.add_argument('--max_tokens', type=int, default=150, 
                        help='Máximo de tokens a generar')
    
    args = parser.parse_args()
    
    if args.check:
        check_dependencies()
    elif args.train:
        if check_dependencies():
            train_model()
    elif args.demo:
        demo()
    elif args.generate:
        model, tokenizer = load_finetuned_model()
        result = generate_text(model, tokenizer, args.generate,
                              max_new_tokens=args.max_tokens,
                              temperature=args.temperature)
        print(f"Generado: {result}")
    else:
        demo()
