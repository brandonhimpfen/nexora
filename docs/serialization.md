# Serialization

Nexora saves state dictionaries as a ZIP container containing JSON metadata and NumPy tensor data. Loading uses `allow_pickle=False`; arbitrary Python object execution is not part of the state format.
