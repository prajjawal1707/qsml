"""QSML language parser.

Converts simple QSML instructions into validated operations.
"""

import re


class QSMLSyntaxError(Exception):
    """Raised when a QSML instruction is invalid."""


def parse_program(source: str) -> list[dict]:
    """Parse a QSML program into an instruction list."""

    instructions = []

    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()

        # Ignore empty lines and comments
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        command = parts[0].upper()
        args = parts[1:]

        try:
            if command == "REGISTER":
                if len(args) != 1:
                    raise ValueError("REGISTER requires a qubit count")

                count = int(args[0])
                if count < 1:
                    raise ValueError("Qubit count must be positive")

                instructions.append({"command": "REGISTER", "qubits": count})

            elif command == "SUPERPOSITION":
                if args != ["ALL"]:
                    raise ValueError("Expected SUPERPOSITION ALL")

                instructions.append({"command": "SUPERPOSITION", "target": "ALL"})

            elif command == "MARK":
                if len(args) != 1:
                    raise ValueError("MARK requires one bitstring")

                if not re.fullmatch(r"[01]+", args[0]):
                    raise ValueError("MARK requires a binary string")

                instructions.append({"command": "MARK", "state": args[0]})

            elif command == "AMPLIFY":
                if len(args) != 1:
                    raise ValueError("AMPLIFY requires an iteration count")

                count = int(args[0])
                if count < 0:
                    raise ValueError("Iterations cannot be negative")

                instructions.append({"command": "AMPLIFY", "iterations": count})

            elif command == "MEASURE":
                if len(args) != 1:
                    raise ValueError("MEASURE requires a shot count")

                shots = int(args[0])
                if shots < 1:
                    raise ValueError("Shots must be positive")

                instructions.append({"command": "MEASURE", "shots": shots})

            elif command == "SHOW":
                if args != ["RESULTS"]:
                    raise ValueError("Expected SHOW RESULTS")

                instructions.append({"command": "SHOW"})

            else:
                raise ValueError(f"Unknown command: {command}")

        except ValueError as error:
            raise QSMLSyntaxError(f"Line {line_number}: {error}") from error

    return instructions
