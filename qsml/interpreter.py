
"""QSML interpreter: executes parsed QSML programs."""

from .language import parse_program, QSMLSyntaxError
from .possibility import grover_search


def execute_program(source: str, seed: int | None = None) -> dict:
    instructions = parse_program(source)

    num_qubits = None
    marked_states = []
    iterations = None
    shots = 1000
    result = None

    for instruction in instructions:
        command = instruction["command"]

        if command == "REGISTER":
            num_qubits = instruction["qubits"]
            marked_states = []
            result = None

        elif command == "SUPERPOSITION":
            if num_qubits is None:
                raise QSMLSyntaxError("REGISTER must come first")

        elif command == "MARK":
            if num_qubits is None:
                raise QSMLSyntaxError("REGISTER must come first")

            state = instruction["state"]

            if len(state) != num_qubits:
                raise QSMLSyntaxError(
                    "MARK bitstring length must match REGISTER"
                )

            marked_states.append(state)

        elif command == "AMPLIFY":
            iterations = instruction["iterations"]

        elif command == "MEASURE":
            shots = instruction["shots"]

            if num_qubits is None or not marked_states:
                raise QSMLSyntaxError(
                    "REGISTER and MARK are required before MEASURE"
                )

            result = grover_search(
                num_qubits=num_qubits,
                marked_states=marked_states,
                shots=shots,
                iterations=iterations,
                seed=seed,
            )

        elif command == "SHOW":
            if result is None:
                raise QSMLSyntaxError(
                    "MEASURE must run before SHOW RESULTS"
                )

    return {
        "qubits": num_qubits,
        "marked_states": marked_states,
        "result": result,
    }
