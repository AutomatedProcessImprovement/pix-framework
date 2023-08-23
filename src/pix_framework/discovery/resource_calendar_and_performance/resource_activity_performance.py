from dataclasses import dataclass
from typing import List


@dataclass
class ResourceDistribution:
    """Resource is the item of activity-resource duration distribution for Prosimos."""

    resource_id: str
    distribution: dict

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {"resource_id": self.resource_id} | self.distribution

    @staticmethod
    def from_dict(resource_distribution: dict) -> "ResourceDistribution":
        return ResourceDistribution(
            resource_id=resource_distribution["resource_id"],
            distribution={key: resource_distribution[key] for key in resource_distribution if key != "resource_id"},
        )


@dataclass
class ActivityResourceDistribution:
    """Activity duration distribution per resource for Prosimos."""

    activity_id: str
    activity_resources_distributions: List[ResourceDistribution]

    def to_dict(self) -> dict:
        """Dictionary with the structure compatible with Prosimos:"""
        return {
            "task_id": self.activity_id,
            "resources": [resource.to_dict() for resource in self.activity_resources_distributions],
        }

    @staticmethod
    def from_dict(activity_resource_distribution: dict) -> "ActivityResourceDistribution":
        return ActivityResourceDistribution(
            activity_id=activity_resource_distribution["task_id"],
            activity_resources_distributions=[
                ResourceDistribution.from_dict(resource_distribution)
                for resource_distribution in activity_resource_distribution["resources"]
            ],
        )
