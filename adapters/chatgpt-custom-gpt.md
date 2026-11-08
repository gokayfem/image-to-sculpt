# ChatGPT (Custom GPT)

1. Create a GPT and paste `adapters/universal-system-prompt.md` (the part below the line) into **Instructions**.
2. Upload as **Knowledge**: `skills/image-to-sculpt/SKILL.md` and the files in `skills/image-to-sculpt/references/`.
3. Enable **Code Interpreter** so it can run the plain-Python parts of `scripts/i2s/` (registration, hull, fit,
   checks) on maps you upload. fal calls and Blender run on your machine.
