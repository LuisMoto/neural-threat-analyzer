import customtkinter as ctk
from inference_engine import predict_threat
from preprocessing import security_preprocess
from features import extract_features


def analyze_input():
    raw_text = textbox.get("1.0", "end-1c")

    if not raw_text.strip():
        result_label.configure(text="STATUS: Awaiting Payload...", text_color="gray")
        prob_label.configure(text="")
        return

    # DEBUG
    clean = security_preprocess(raw_text)
    feats = extract_features(clean)
    print(f"\n--- CLEAN TEXT ---\n{clean}")
    print(f"\n--- FEATURES ---\n{feats}")

    result = predict_threat(raw_text)
    prediction = result["prediction"]

    if prediction == 0:
        color = "#00FF00"
    elif prediction == 1:
        color = "#FFCC00"
    elif prediction == 2:
        color = "#FF0044"
    else:
        color = "gray"

    result_label.configure(
        text=f"DETECTED: {result['threat_name']}\nCONFIDENCE: {result['confidence']:.2f}%",
        text_color=color
    )

    if "probabilities" in result:
        prob_lines = "  |  ".join(
            f"{name}: {prob:.1f}%"
            for name, prob in result["probabilities"].items()
        )
        prob_label.configure(text=prob_lines)
    else:
        prob_label.configure(text="")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("700x520")
app.title("Neural Threat Analyzer v2.0")

title = ctk.CTkLabel(app, text="NEURAL THREAT ANALYZER", font=("Courier", 24, "bold"), text_color="#00FF00")
title.pack(pady=(30, 5))

subtitle = ctk.CTkLabel(app, text="Input text payload or email content for NLP classification", font=("Courier", 11))
subtitle.pack(pady=(0, 15))

textbox = ctk.CTkTextbox(app, width=600, height=160, font=("Courier", 13), fg_color="#1E1E1E", border_color="#00FF00", border_width=1)
textbox.pack(pady=8)

analyze_btn = ctk.CTkButton(app, text="INITIALIZE SCAN", font=("Courier", 16, "bold"), command=analyze_input, fg_color="#005500", hover_color="#00AA00")
analyze_btn.pack(pady=16)

result_label = ctk.CTkLabel(app, text="STATUS: Awaiting Payload...", font=("Courier", 18, "bold"))
result_label.pack(pady=8)

prob_label = ctk.CTkLabel(app, text="", font=("Courier", 10), text_color="#888888")
prob_label.pack(pady=4)

if __name__ == "__main__":
    app.mainloop()