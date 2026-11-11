"""Build A/B/C from one frozen catalog, grid and survivor policy."""

from dataclasses import replace

from .config import Arm


def policy_for_arm(arm, policy, family_fraction):
    from fedot_ind.core.optimizer.search_policy.contracts import MutationMix
    arm = Arm(arm)
    stages = tuple(replace(stage, family_fraction=family_fraction) for stage in policy.stages)
    if arm is not Arm.STAGED:
        joint = next(stage for stage in stages if stage.name == 'joint')
        if arm is Arm.BASELINE:
            weights = joint.weights.to_record()
            weights['rewrite'] = 0.
            total = sum(weights.values())
            if total <= 0:
                raise ValueError('Baseline requires non-rewrite actions')
            joint = replace(joint, weights=MutationMix(**{key: value / total for key, value in weights.items()}))
        stages = (replace(joint, name=arm.value, until=1.),)
    return replace(policy, stages=stages)
