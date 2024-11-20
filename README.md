# Priceforge

`priceforge` is a financial derivatives pricing library that supports multiple models and pricing engines. This guide will help you get started with basic option pricing.
## Install
In your terminal run:
```bash
pip install priceforge
```

## Getting Started



### 1. Creating an Option

Create a vanilla option by specifying expiry date, strike price, and option type:

```python
from priceforge.api import create_option

# Create a call option expiring March 1st, 2024 with strike price 100
option = create_option("2024-03-01", 100.0, "CALL")

# For options on futures, specify the underlying expiry
futures_option = create_option(
    "2024-03-01",    # Option expiry
    100.0,           # Strike price
    "CALL",          # Option type
    underlying_expiry="2024-03-03"  # Futures expiry
)
```

### 2. Setting Up a Model

Choose from several pricing models, each with configurable parameters:

```python
from priceforge.api import Model

# Create a Black-Scholes model with default parameters
model = Model("BLACK_SCHOLES", spot={"value": 90.0, "volatility": 0.16})

# Configure model parameters
model.update_config(spot={"value": 100.0})

# View current configuration
config = model.get_config()
# Returns: {
#     "spot": {"value": 100.0, "volatility": 1.0},
#     "rate": {"value": 0.0}
# }
```

Available models:
- `BLACK_SCHOLES` - Classic Black-Scholes model
- `BLACK_76` - Black 1976 model for futures options
- `HESTON` - Heston stochastic volatility model
- `TROLLE_SCHWARTZ` - Trolle-Schwartz model for commodity options

### 3. Choosing a Pricing Engine

Select and configure a pricing engine:

```python
from priceforge.api import Engine

# Available engines with example configurations
closed_form = Engine("CLOSED_FORM")
monte_carlo = Engine("MONTE_CARLO", n_paths=100_000)
fourier = Engine("FOURIER", config={
    "integral_truncation": 100,
    "dampening_factor": 0.75
})

# Update engine configuration
fourier.update_config(dampening_factor=1.0)

# View current configuration
config = fourier.get_config()
# Returns: {
#     "method": "CARR_MADAN",
#     "integral_truncation": 100,
#     "dampening_factor": 0.75
# }
```

Available engines:
- `CLOSED_FORM` - Analytical solutions (when available)
- `MONTE_CARLO` - Monte Carlo simulation
- `FOURIER` - Fourier transform methods

### 4. Pricing Options

Combine all components to price an option:

```python
# Set up components
valuation_time = "2024-02-01"
option = create_option("2024-03-01", 100.0, "CALL")
model = Model("BLACK_SCHOLES")
engine = Engine("CLOSED_FORM")

# Calculate price
price = engine.price(valuation_time, option, model)
```

## Model and Engine Compatibility

Different combinations of models and engines are supported:

| Model              | Closed Form | Monte Carlo | Fourier |
|-------------------|-------------|-------------|---------|
| BLACK_SCHOLES     | ✓           | ✓           | ✓       |
| BLACK_76          | ✓           | ✓           | ✓       |
| HESTON            | ✗           | ✓           | ✓       |
| TROLLE_SCHWARTZ   | ✗           | ✗           | ✓       |


## Error Handling

The library includes built-in validation:

```python
# Invalid engine configuration raises ValueError
Engine("CLOSED_FORM", config={"not_existing_key": 1})  # ValueError

# Invalid engine type raises ValueError
Engine("UNKNOWN")  # ValueError
```
