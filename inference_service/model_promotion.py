"""Utilities for promoting and retrieving models in ClearML.

This module uses ClearML *tags* to mark which model is considered production
(`prod`) vs. a pre-production candidate (`candidate`), and relies on ClearML's
published model registry for discoverability.
"""

from clearml import Model
from clearml.backend_api.session.client import APIClient


def promote_model(model_id: str, new_tag: str = "prod", old_tag: str = "candidate") -> None:
    """Promote a ClearML model by swapping tags and publishing it.

    The default flow is:
    1) Remove the `candidate` tag (if present).
    2) Add the `prod` tag.
    3) Publish the model so it shows up in the published model registry.
    """
    model = Model(model_id=model_id)

    # Update tags via the backend API to ensure the change is persisted.
    # Note: `list.append(...)` returns `None`, so we build the final list first.
    existing_tags = model.tags or []
    new_model_tags = [tag for tag in existing_tags if tag != old_tag]
    if new_tag not in new_model_tags:
        new_model_tags.append(new_tag)

    client = APIClient()
    client.models.edit(model=model_id, tags=new_model_tags)

    # Publish the model (required for `only_published=True` queries).
    model.publish()


def get_prod_model(model_name: str, project_name: str = "Engagement Prediction") -> Model:
    """Return the single published production model for a given name/project."""
    published_prod_models = Model.query_models(
        project_name=project_name,
        model_name=model_name,
        only_published=True,
        tags=["prod"],
    )

    if len(published_prod_models) > 1:
        raise ValueError(f"Multiple published prod models found for model name '{model_name}'!")
    if len(published_prod_models) == 0:
        raise ValueError(f"No published prod models found for model name '{model_name}'!")

    return published_prod_models[0]
