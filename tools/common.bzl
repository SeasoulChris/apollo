# Sanitize a dependency so that it works correctly from code that includes
# apollo-fuel as a submodule.
def clean_dep(dep):
    return str(Label(dep))
