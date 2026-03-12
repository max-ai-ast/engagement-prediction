from clearml import Model
from clearml.backend_api.session.client import APIClient

if __name__ == "__main__":
    
    client = APIClient()

    result_models = Model.query_models(model_name="test_model", include_archived=True) # only_published=True
    
    for i, model in enumerate(result_models):
        print("=" * 80)
        print(f"MODEL {i}:")
        print("=" * 80)
        print(f"Model ID: {model.id}\n")
        
        for k, v in model.get_all_metadata().items():
            print(f"{k}: {v}")
        
        print(f"\nSystem Tags: {model.system_tags}")
        is_archived = model.archived_tag in model.system_tags
        print(f"Archived: {is_archived}")
        if is_archived:
            model.unarchive()
            print(f"Unarchived model! (I think)")

        # Can update a model's tags via the low-level API client:
        tag = "test_tag_tres"
        print(f"Tags before: {model.tags}")
        print(f"Adding tag: {tag}...")
        client.models.edit(tags=model.tags + [tag], model=model.id)
        model = Model(model_id=model.id)
        print(f"Tags after: {model.tags}")
        
        print("\n")