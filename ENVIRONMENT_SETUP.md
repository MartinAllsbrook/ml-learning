# ML Learning Environment Setup

This repository uses a Python virtual environment to manage dependencies. Follow these steps to set up the same environment on your machine.

## Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ml-learning
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell/CMD):**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   python -c "import numpy, matplotlib; print('✅ Environment ready!')"
   ```

## Working with Jupyter Notebooks

After setting up the environment, you can run the notebooks:

```bash
jupyter notebook
```

Or if using VS Code, make sure to select the correct Python interpreter (`.venv/Scripts/python.exe` on Windows).

## Adding New Dependencies

When you install new packages:

1. **Install the package:**
   ```bash
   pip install package-name
   ```

2. **Update requirements.txt:**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Commit the updated requirements.txt:**
   ```bash
   git add requirements.txt
   git commit -m "Add package-name dependency"
   ```

## Environment Details

- **Python Version:** 3.13.5
- **Key Packages:**
  - NumPy 2.3.1 - Numerical computing
  - Matplotlib 3.10.3 - Data visualization
  - (Add other packages as you install them)

## Troubleshooting

- **Module not found errors:** Make sure your virtual environment is activated
- **Permission errors:** On Windows, you might need to run PowerShell as administrator
- **Package conflicts:** Delete `.venv` folder and recreate the environment from scratch

## Team Collaboration

- ✅ **DO commit:** `requirements.txt`, source code, notebooks
- ❌ **DON'T commit:** `.venv/` folder, `__pycache__/`, `.ipynb_checkpoints/`

Everyone on the team should use the same `requirements.txt` to ensure consistent environments.
