# Contributing to soweak

First off, thank you for considering contributing to `soweak`. It's people like you that make `soweak` such a great tool.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Code of Conduct

This project and everyone participating in it is governed by the [soweak Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

This is a great way to contribute. Before creating a bug report, please check that the issue hasn't already been reported. If you've found a new bug, please create an issue that includes:

- A clear title and description.
- The version of `soweak` you are using.
- A code sample or an executable test case demonstrating the bug.
- The expected behavior and what actually happened.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please create an issue that includes:

- A clear title and description.
- A step-by-step description of the suggested enhancement in as many details as possible.
- Use cases that this enhancement would enable.

## Contribution Workflow

### Development Setup

To get started with development, you'll need to set up a local environment.

1.  Fork the repository on GitHub.
2.  Clone your fork locally:
    ```bash
    git clone https://github.com/your-username/soweak.git
    cd soweak
    ```
3.  Create a virtual environment and install the dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -e ".[dev]"
    ```

### Pull Request Process

1.  Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-amazing-feature
    ```
2.  Make your changes. Ensure you adhere to the coding style.
3.  Run the tests to ensure everything is working correctly:
    ```bash
    pytest
    ```
4.  Commit your changes with a clear commit message.
5.  Push your branch to your fork:
    ```bash
    git push origin feature/your-amazing-feature
    ```
6.  Open a pull request from your fork to the main `soweak` repository. Provide a clear description of the changes you've made.

### Coding Style

- We use **Black** for code formatting and **isort** for import sorting. Before committing, please run these tools to format your code:
  ```bash
  black .
  isort .
  ```
- We use **mypy** for static type checking. Please ensure your code passes type checking:
  ```bash
  mypy soweak
  ```

### Testing

Please add tests for any new features or bug fixes. `soweak` uses `pytest`. You can run the full test suite with:

```bash
pytest
```

## License

By contributing to `soweak`, you agree that your contributions will be licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
