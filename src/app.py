import customtkinter as ctk
from inference_engine import predict_threat

def analyze_input():
    # Get text from the view
    raw_text = textbox.get("1.0", "end-1c")
    
    if not raw_text.strip():
        result_label.configure(text="STATUS: Awaiting Payload...", text_color="gray")
        return
    
    # Query the Inference Engine (Separated logic)
    result = predict_threat(raw_text)
    
    # Update the View based on the response
    prediction = result["prediction"]
    
    # Assign alert colors
    if prediction == 0:
        color = "#00FF00" # Green (Safe)
    elif prediction == 1:
        color = "#FFCC00" # Yellow (Phishing)
    elif prediction == 2:
        color = "#FF0044" # Red (SQLi)
    else:
        color = "gray"
        
    result_label.configure(
        text=f"DETECTED: {result['threat_name']}\nCONFIDENCE: {result['confidence']:.2f}%", 
        text_color=color
    )

# GUI Construction
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green") 

app = ctk.CTk()
app.geometry("650x450")
app.title("Neural Threat Analyzer v1.0")

# Visual Elements
title = ctk.CTkLabel(app, text="NEURAL THREAT ANALYZER", font=("Courier", 24, "bold"), text_color="#00FF00")
title.pack(pady=(30, 10))

subtitle = ctk.CTkLabel(app, text="Input text payload or email content for NLP classification", font=("Courier", 12))
subtitle.pack(pady=(0, 20))

textbox = ctk.CTkTextbox(app, width=550, height=150, font=("Courier", 14), fg_color="#1E1E1E", border_color="#00FF00", border_width=1)
textbox.pack(pady=10)

analyze_btn = ctk.CTkButton(app, text="INITIALIZE SCAN", font=("Courier", 16, "bold"), command=analyze_input, fg_color="#005500", hover_color="#00AA00")
analyze_btn.pack(pady=20)

result_label = ctk.CTkLabel(app, text="STATUS: Awaiting Payload...", font=("Courier", 18, "bold"))
result_label.pack(pady=10)

if __name__ == "__main__":
    app.mainloop()