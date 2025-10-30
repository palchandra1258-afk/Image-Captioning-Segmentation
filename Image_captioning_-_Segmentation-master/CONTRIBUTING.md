# Contributing to Image Captioning & Segmentation

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- **Clear title**: Describe the issue briefly
- **Steps to reproduce**: How to trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, GPU/CPU
- **Screenshots**: If applicable

### Suggesting Features

For new features:
- Check existing issues to avoid duplicates
- Describe the feature and its benefits
- Explain how it fits with the project goals
- Provide examples or mockups if possible

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Write/update tests** for your changes
5. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/ -v
   ```
6. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/demos if applicable

## 🎯 Development Setup

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/image-caption-seg.git
cd image-caption-seg
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks before commits.

### 3. Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test file
pytest tests/test_viz.py -v
```

### 4. Lint and Format

```bash
# Check code style
flake8 . --max-line-length=120

# Format code
black .

# Sort imports
isort .
```

## 📝 Code Style Guidelines

### Python Code

- **PEP 8**: Follow Python style guide
- **Line length**: Maximum 120 characters
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type annotations where helpful
- **Comments**: Explain "why", not "what"

#### Example:

```python
def process_image(image: Image.Image, device: str = 'cpu') -> Dict[str, Any]:
    """
    Process an image for model inference.
    
    Args:
        image: PIL Image object
        device: Computation device ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing processed results
        
    Raises:
        ValueError: If image is invalid
    """
    # Implementation here
    pass
```

### Streamlit Code

- **Session state**: Use `st.session_state` for persistence
- **Caching**: Apply `@st.cache_resource` for models
- **User feedback**: Provide progress bars and status messages
- **Error handling**: Use `st.error()` for user-friendly errors

### Testing

- **Test coverage**: Aim for >80% coverage
- **Test naming**: `test_<function_name>_<scenario>`
- **Fixtures**: Use pytest fixtures for reusable test data
- **Assertions**: One assertion concept per test

#### Example:

```python
def test_overlay_instance_masks_with_valid_input(sample_image, sample_masks):
    """Test that overlay_instance_masks works with valid inputs."""
    result = overlay_instance_masks(sample_image, sample_masks)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape
```

## 🗂️ Project Structure

When adding new features, follow this structure:

```
├── models/              # Model architectures and wrappers
├── inference/           # Inference pipelines
├── utils/               # Utility functions
│   ├── viz.py          # Visualization
│   ├── coco_utils.py   # Dataset utilities
│   └── io.py           # I/O operations
├── static/              # Static assets (CSS, images)
├── tests/               # Test files (mirror source structure)
└── docs/                # Documentation
```

## 🔧 Adding New Features

### Adding a New Model

1. **Add model to manifest**: Update `models_manifest.json`
2. **Create wrapper**: Add to `models/wrappers.py`
3. **Update pipeline**: Modify relevant inference pipeline
4. **Add to UI**: Update app.py dropdown options
5. **Write tests**: Add test cases
6. **Document**: Update README with model info

### Adding New Functionality

1. **Plan the feature**: Consider architecture and dependencies
2. **Write tests first**: TDD approach preferred
3. **Implement**: Follow code style guidelines
4. **Test thoroughly**: Unit and integration tests
5. **Document**: Add docstrings and update README
6. **Create PR**: With clear description

## 🧪 Testing Guidelines

### Unit Tests

- Test individual functions in isolation
- Mock external dependencies
- Cover edge cases and error conditions

### Integration Tests

- Test complete workflows
- Use realistic data
- Verify outputs match expectations

### Test Organization

```
tests/
├── test_models.py          # Model wrapper tests
├── test_inference.py       # Pipeline tests
├── test_viz.py             # Visualization tests
├── test_coco_utils.py      # Dataset utility tests
├── test_io.py              # I/O utility tests
└── conftest.py             # Shared fixtures
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short description.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception occurs
        
    Example:
        >>> function_name(value1, value2)
        expected_output
    """
```

### README Updates

When adding features:
- Update the features list
- Add usage examples
- Document configuration options
- Update architecture diagram if needed

## 🐛 Debugging Tips

### Common Issues

1. **Import errors**: Ensure `__init__.py` files exist
2. **CUDA errors**: Check GPU availability and memory
3. **Model loading**: Verify checkpoint paths
4. **Test failures**: Run with `-v` flag for details

### Debugging Tools

```bash
# Run with debugger
python -m pdb app.py

# Verbose test output
pytest tests/ -vv -s

# Show print statements
pytest tests/ -s
```

## 🚀 Release Process

1. **Update version**: In `setup.py` or `__version__.py`
2. **Update CHANGELOG**: Document changes
3. **Run full test suite**: Ensure all tests pass
4. **Create release branch**: `release/v1.x.x`
5. **Tag release**: `git tag v1.x.x`
6. **Build Docker images**: Update tags
7. **Deploy to staging**: Test in staging environment
8. **Merge to main**: After validation
9. **Deploy to production**: Follow deployment guide

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] No linting errors
- [ ] Commit messages are clear
- [ ] PR description explains changes
- [ ] Related issues are referenced

## 🎓 Learning Resources

### Project-Related
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [COCO Dataset](https://cocodataset.org/)

### Best Practices
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Git Best Practices](https://www.git-scm.com/book/en/v2)

## 🏆 Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Acknowledged in commit messages

## 📧 Contact

For questions or discussions:
- Open a GitHub issue
- Email: [maintainer@email.com]
- Discord: [project-discord-link]

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.

---

**Thank you for contributing!** 🎉

Your efforts help make this project better for everyone.
