from cadello import Analyser

w = Analyser()  # Wren 1.5 (default)
print("Wren 1.5 ready. Type a review, Ctrl+C to quit.")
while True:
    try:
        t = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        break
    if t.strip():
        p = w.predict_proba(t)[0]
        s = int(p.argmax()) + 1
        print(f"{'*' * s:<5}  {w.label(s)}   " + "  ".join(f"{i}*:{x:.0%}" for i, x in enumerate(p, 1)))
