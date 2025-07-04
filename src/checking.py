import torch
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer

def load_and_validate_merged_gptq(model_path: str):
    """
    Lädt ein bereits gemergtes GPTQ-Modell, das FP16-qzeros enthält.

    Args:
        model_path (str): Der Pfad zum Verzeichnis des gemergten Modells.
    """
    print(f"--- Lade gemergtes GPTQ-Modell: {model_path} ---")

    # 1. Wir müssen eine GPTQConfig bereitstellen, um transformers mitzuteilen,
    #    dass es sich um die GPTQ-Architektur handelt. Die Werte hier sind
    #    Standardwerte und werden teilweise durch die config.json im Modell überschrieben.
    gptq_config = GPTQConfig(bits=4, use_exllama=False)

    try:
        # 2. Lade das Modell mit der GPTQConfig UND dem entscheidenden Flag.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=gptq_config,
            device_map="auto",
            torch_dtype=torch.float16,
            # DIES IST DIE LÖSUNG:
            # Erlaube dem Lader, Tensoren zu akzeptieren, deren Form nicht mit der
            # initial erstellten Modell-Hülle übereinstimmt.
            ignore_mismatched_sizes=True,
        )
        print("Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return

    # 3. Validierung, um zu beweisen, dass es funktioniert hat.
    layer_path_str = "model.layers[0].self_attn.q_proj"
    print(f"\n--- Inspektion der Schicht: {layer_path_str} ---")
    target_layer = model.get_submodule(layer_path_str.replace("model.", ""))

    if not hasattr(target_layer, "qzeros"):
        print("Die Schicht hat kein 'qzeros'-Attribut.")
        return

    qzeros_tensor = target_layer.qzeros
    print(f"\n[Validierungs-Ergebnis]")
    print(f"Datentyp (dtype) von qzeros: {qzeros_tensor.dtype}")
    print(f"Form (shape) von qzeros:    {qzeros_tensor.shape}")

    if qzeros_tensor.dtype == torch.float16 and qzeros_tensor.shape[1] > 256:
        print("\n✅ VALIDIERUNG ERFOLGREICH: Das Modell wurde als GPTQ-Modell mit entpackten FP16-qzeros geladen.")
    else:
        print("\n❌ VALIDIERUNG FEHLGESCHLAGEN: Die qzeros sind nicht im erwarteten FP16-Format.")


if __name__ == "__main__":
    # --- HIER ANPASSEN ---
    # Geben Sie hier den Pfad zu Ihrem Verzeichnis mit dem gemergten Modell an.
    MERGED_MODEL_PATH = "/home/gap/Documents/peft/qalora_output_merged_model_via_beta_shift"

    load_and_validate_merged_gptq(model_path=MERGED_MODEL_PATH)