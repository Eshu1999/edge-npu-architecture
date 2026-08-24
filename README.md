# Pheneon NPU Architecture & Compiler

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Architectural%20Specification-teal.svg)]()
[![Target: ASIC / MPW](https://img.shields.io/badge/Target-Silicon%20Prototyping-blue.svg)]()

> **Pheneon** is an application-specific hardware accelerator and compilation framework engineered to eliminate the von Neumann memory wall in deep learning inference. 

---

## Overview

Modern neural network workloads are heavily bottlenecked by memory bandwidth and dynamic power dissipation when running on traditional CPU/GPU architectures. Pheneon solves this by coupling a custom 2D spatial systolic array with an automated PyTorch-to-hardware compilation pipeline (`pytorch_compiler.py` and `npu_bridge.py`).

By keeping core register weights stationary and streaming feature map activations locally across processing elements (PEs), Pheneon achieves ultra-low latency inference and high arithmetic intensity at the edge.

---

## Core Architecture Specifications

| Architectural Parameter | Specification Details |
| :--- | :--- |
| **Microarchitecture** | Scalable 2D Systolic Array (Configurable $16 \times 16$ to $64 \times 64$ PEs) |
| **Data Precision** | Mixed-precision execution (INT8, INT16, and dynamic FP16 scaling) |
| **Memory Hierarchy** | Local Scratchpad SRAM with distributed double-buffering |
| **Interconnect Fabric** | Low-latency AXI-Stream interface for direct tensor streaming |
| **Target Technology** | TSMC Mature / Advanced Nodes (Targeting MPW Shuttle Prototyping) |

---

## Theoretical Model & Dataflow

Pheneon implements a **Weight-Stationary (WS)** dataflow model. Weights are loaded once into local PE register files, minimizing external memory fetches:

* **Systolic Accumulation:** 
  $$\mathbf{Y}_{i,j} = \sum_{k=1}^{K} A_{i,k} \times B_{k,j}$$
  *(where $A$ represents streaming activations and $B$ represents stationary weights)*

* **Quantization Mapping:** 
  Automated INT8 mapping bounds accuracy loss using symmetric quantization scaling factors ($S$):
  $$q = \text{clip}\left( \left\lfloor \frac{x}{S} \rceil + Z, \, -128, \, 127 \right)$$

---

## Software Compiler & Pipeline

The Pheneon framework translates high-level PyTorch models into optimized hardware schedules through four automated phases:
1. **Graph Ingestion & Parsing:** Extracts Directed Acyclic Graphs (DAG) from TorchScript / ONNX IR.
2. **Layer Fusion:** Merges sequential operators (e.g., Conv2D + ReLU + Quantization) to eliminate memory round-trips.
3. **Hardware Scheduling:** Allocates tensor tiles across physical processing blocks to maximize data reuse.
4. **Runtime Execution:** Emits instruction bundles executed seamlessly via the `npu_bridge` interface.

---

## Projected Performance Benchmarks

| Benchmark / Metric | Standard GPU Baseline | Pheneon NPU Architecture |
| :--- | :--- | :--- |
| **ResNet-50 Inference Latency** | 12.4 ms | **3.1 ms** *(4.0x Speedup)* |
| **Power Efficiency** | 1.8 TOPS/W | **14.2 TOPS/W** *(7.8x Gain)* |
| **Memory Bottleneck** | DRAM Bandwidth Bound | **Minimized via SRAM Scratchpads** |

---

## Documentation & Repository Structure

* [`Pheneon_NPU_Architecture_Specification.docx`](Pheneon_NPU_Architecture_Specification.docx) — Comprehensive technical documentation, mathematical models, and architectural breakdown.
* **Core RTL & Source Code:** Maintained in secure, private repositories to protect proprietary microarchitecture and intellectual property.

---

## License & Intellectual Property

All underlying hardware description logic (SystemVerilog RTL), systolic array layouts, and compiler source code are the **proprietary intellectual property of Pheneon**. Unauthorized copying, simulation, or commercial reproduction is strictly prohibited.
