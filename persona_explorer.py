"""Compatibility entry point for the Persona Explorer service."""
import importlib
import sys

import persona

if __name__ == "__main__":
    importlib.import_module("persona_explorer").main()
else:
    # Preserve imports of the former root module, including private attributes.
    sys.modules[__name__] = importlib.import_module("persona.persona_explorer")
