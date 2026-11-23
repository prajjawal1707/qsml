from psutil import virtual_memory
from .hardware import inspect_hardware
from .resources import state_vector_bytes, format_bytes, recommend_qubit_limit
from .possibility import grover_search


def main() -> None:
    info = inspect_hardware()
    print("=" * 62)
    print("QSML 0.3 — POSSIBILITY ENGINE")
    print("=" * 62)
    print("\nHARDWARE")
    print(f"OS: {info['system']} {info['release']}")
    print(f"CPU: {info['processor']}")
    print(f"Cores: {info['physical_cores']} physical / {info['logical_cores']} logical")
    print(
        f"RAM: {info['ram_total_gib']} GiB total / {info['ram_available_gib']} GiB available"
    )
    print(f"Disk: {info['disk_free_gib']} GiB free of {info['disk_total_gib']} GiB")
    print("Display adapters:", "; ".join(info["display_adapters"]))
    print("NVIDIA:", "; ".join(info["nvidia_smi_gpus"]))

    available = virtual_memory().available
    limit = recommend_qubit_limit(available)
    print("\nMEMORY PLANNER")
    for n in (10, 20, 23, 25, 30):
        print(f"{n:2d} qubits: {format_bytes(state_vector_bytes(n))} raw state vector")
    print(
        f"Conservative estimate now: about {limit} qubits (includes working-space allowance)"
    )
    print("This is a guardrail estimate, not a guaranteed safe allocation.")

    print("\nPOSSIBILITY ENGINE — GROVER DEMO")
    print("Search space: all 3-bit strings")
    print("Marked answer: 101")
    result = grover_search(3, ["101"], shots=1000, seed=42)
    print(f"Amplitude-amplification iterations: {result.iterations}")
    print("Top candidates by probability:")
    for bitstring, probability in result.top_candidates:
        print(f"  {bitstring}: {probability:.3%}")
    print("Measurement counts:", result.measurements)
    print("\nQSML simulates quantum algorithms using classical CPU/RAM.")
    print("It does not provide physical quantum speedup.")


if __name__ == "__main__":
    main()
