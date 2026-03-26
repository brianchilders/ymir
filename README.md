# 🛡️ HEIMDALL  
### A Local-First Ambient Intelligence Home Prototype

> **Most AI today lives in apps. Ambient intelligence should live in the environment.**

---

## Overview

**Heimdall** is a working prototype exploring what ambient intelligence looks like when it is:

- **Local-first**
- **Privacy-preserving**
- **Memory-enabled**
- **Embedded in the home—not in an app**

This project integrates:

- 🟢 **OpenHome speakers** → ambient interface layer  
- 🟡 **Home Assistant** → device + sensor orchestration  
- 🔵 **memory-mcp** → persistent context + memory layer  
- 🔴 **Heimdall nodes** → room-level AI presence (voice + awareness)  

---

## 🎯 The Goal

This is **not** an attempt to build a fully autonomous AI home.

Instead, Heimdall explores a simpler, more important question:

> Can a home become meaningfully more helpful if it remembers, understands context, and responds locally over time?

---

## 🧠 Core Capabilities

### 1. A Home That Remembers
Using **memory-mcp**, the system stores:

- preferences (temperature, lighting, routines)  
- environmental patterns  
- prior interactions  
- contextual signals from the home  

➡️ The home accumulates intelligence instead of resetting every day.

---

### 2. Ambient Interaction (Not Commands)

With **OpenHome speakers + Heimdall nodes**:

- voice is **context-aware**, not command-driven  
- responses are grounded in:
  - current home state  
  - remembered patterns  
  - recent interactions  

➡️ Interaction becomes **ambient**, not transactional.

---

### 3. Environmental Awareness (Bounded)

A lightweight perception layer introduces:

- event-based object detection  
- no continuous surveillance model  

Examples:
- `package_detected`
- `room_occupied`
- `garage_state_changed`

➡️ Vision becomes **input**, not monitoring.

---

## 🏗️ Architecture

```
                ┌──────────────────────┐
                │   OpenHome Speaker   │
                │ (Ambient Interface)  │
                └─────────┬────────────┘
                          │
                ┌─────────▼────────────┐
                │   Heimdall Node       │
                │ (Voice + Context)     │
                └─────────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼────────┐ ┌─────▼─────────┐
│ Home Assistant│ │  memory-mcp     │ │  AI/Agent     │
│ (Devices)     │ │ (Memory Layer)  │ │ (Reasoning)   │
└───────────────┘ └────────────────┘ └───────────────┘

            Optional: Object Detection → Event Layer
```

---

## 🛠️ Heimdall Device (Prototype)

A **room-level AI node** built on:

- Raspberry Pi 5 (16GB)
- Hailo AI accelerator
- ReSpeaker XVF3800 4-mic array

### Responsibilities

- listens (wake / trigger-based)  
- retrieves context from memory  
- understands current home state  
- routes actions via Home Assistant  
- responds via OpenHome speakers  

> Think: **a local watchman for each room**

---

## 🔄 Example Scenarios

### 🟢 Preference-Aware Comfort
> “It’s evening, and you usually prefer it warmer here.”

System adjusts environment based on learned patterns.

---

### 🟡 Context-Aware Voice
> “Did I leave the garage open?”

Response uses:
- current state  
- recent activity  
- memory context  

---

### 🔵 Intention Carryover
> “Remind me to take that package inside when I get home.”

System remembers and triggers when conditions match.

---

### 🔴 Event-Based Awareness
> “Package detected at front door.”

Event → memory → optional action or notification

---

## 🔐 Privacy Model

Heimdall is designed with **privacy as a first principle**:

- local processing preferred by default  
- minimal raw data retention  
- event-based perception (not video storage)  
- explainable actions  
- no requirement for cloud dependence  

---

## 📦 Project Deliverables

- Ambient AI **reference architecture**
- Heimdall **prototype device (v1)**
- OpenHome + memory integration model
- Memory-enhanced automation examples
- Object detection pilot (event-based)
- Demonstration scenarios (4+ use cases)
- Open-source release + documentation

---

## 🧪 What This Project Tests

- Does **memory improve automation usefulness**?
- What does **ambient interaction actually feel like**?
- How much intelligence can stay **local and private**?
- What is the **minimum viable perception layer**?

---

## 🌍 Open Source Commitment

All non-sensitive components will be released as open source:

- architecture  
- integration patterns  
- code  
- findings  
- lessons learned  

The goal is to create a **foundation others can build on**, not a closed system.

---

## 🧭 Why This Matters

Today’s systems:
- react, but don’t remember  
- respond, but don’t understand context  
- automate, but don’t adapt  

Heimdall explores a different path:

> **A home that quietly learns, remembers, and assists—without needing to be asked every time.**

---

## 👤 About

**Brian Childers**  
Builder focused on:
- local-first AI systems  
- memory-driven architectures  
- ambient intelligence in real environments  

---

## 🤝 Collaboration

This project is being developed with alignment toward:

- OpenHome ecosystem  
- ambient intelligence principles  
- local-first, privacy-first design  

If you're exploring similar ideas or want to collaborate:

👉 Open an issue  
👉 Reach out directly  

---

## 🚀 Status

**In active development**

- [x] memory-mcp foundation  
- [x] Home Assistant integration direction  
- [ ] Heimdall v1 prototype  
- [ ] Object detection pilot  
- [ ] Full demo scenarios  

---

## ⭐ Final Thought

> Ambient intelligence shouldn’t feel like using a system.  
> It should feel like the environment understands you.
