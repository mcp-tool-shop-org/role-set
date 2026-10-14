"""role-set: a generator for role-shaping training data.

The plan fixes every slot before a model writes; the guards refuse anything that
isn't a local endpoint, an allowed writer, or a granted card; the gates come from
aspire-si's package and are never copied here.
"""

__version__ = "0.2.0"
