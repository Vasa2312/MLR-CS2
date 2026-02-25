# RBE 577: Physics-Based Learning for Robotic Manipulation
## Programming Assignment 2: Learning Push Dynamics

### Overview
In this assignment, you will implement a physics-based learning framework for planar pushing manipulation. The goal is to combine analytical physics models with learning to predict and plan pushing motions for objects.

### Learning Objectives
- Understand and implement rigid body dynamics for planar pushing
- Combine physics-based models with neural networks
- Build bidirectional models (forward prediction and inverse planning)
- Work with real robotics data

### Project Structure
```
src/
├── config/
│   └── default.yaml      # Configuration file (provided)
├── lib/
│   ├── physics.py        # Physics engine (implement this)
│   └── models.py         # Neural networks (implement this)
├── helpers/
│   ├── utils.py          # Data loading utilities (provided)
│   └── config.py         # Config handling (provided)
└── main.py               # Training script (partially provided)
```



### Getting Started
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Examine the provided utilities in `helpers/`

3. Start with implementing the physics engine:
   - Study planar pushing mechanics
   - Derive motion equations
   - Implement numerical integration

4. Move on to the learning components:
   - Understand the hybrid architecture
   - Implement network layers
   - Connect physics and learning

### Submission Requirements
1. Complete code implementation
2. Project report (5 pages max) including:
   - Mathematical derivations
   - Design decisions
   - Experimental results
   - Analysis
3. Code documentation
4. Test results

### Grading Criteria
- Physics implementation correctness (30%)
- Model architecture and implementation (30%)
- Code quality and documentation (20%)
- Analysis and results (20%)


