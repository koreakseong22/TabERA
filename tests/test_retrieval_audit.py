import unittest

import torch

from libs.retrieval_audit import classify_retrieval_branches, instrument_retrieval
from libs.tabera import MemoryBank
from tests.test_reproduction_state import ReproductionStateTests


class RetrievalAuditTests(unittest.TestCase):
    def test_all_branch_classifications(self):
        memory = type("Memory", (), {})()
        memory.filled = torch.tensor(20)
        memory._cached_groups = torch.zeros(3, 10, dtype=torch.long)
        memory._cached_group_sizes = torch.tensor([10, 3, 2])
        memory._cached_extended = torch.zeros(3, 12, dtype=torch.long)
        memory._cached_extended_sizes = torch.tensor([12, 9, 6])
        assignment = torch.tensor([0, 1, 2])
        records = classify_retrieval_branches(memory, 3, 8, assignment)
        self.assertEqual([r["fallback_type"] for r in records],
                         ["none", "adjacent_region", "global"])
        # Training self-exclusion raises the initial candidate requirement.
        memory._cached_group_sizes[0] = 8
        self.assertEqual(classify_retrieval_branches(
            memory, 1, 8, assignment[:1], torch.tensor([0]))[0]["fallback_type"],
            "adjacent_region")
        memory._cached_groups = None
        self.assertEqual(classify_retrieval_branches(
            memory, 1, 8, assignment[:1])[0]["fallback_type"], "global")

    def test_branch_classifier_matches_real_retrieve_candidate_pools(self):
        """Make each branch's candidate pool distinguishable in returned IDs."""
        memory = MemoryBank(max_size=12, embed_dim=2)
        memory.filled.fill_(12)
        memory.ptr.fill_(0)
        # All keys are finite/nonzero. Exact ranking is irrelevant for the
        # first two rows because their selected pools contain exactly k IDs.
        keys = torch.tensor([
            [1.00, .01], [.99, .02], [.98, .03],       # group 0: none
            [.01, 1.00], [.02, .99],                   # group 1: own
            [.40, .60],                                # group 2: own
            [.03, .98],                                # group 1: adjacent
            [.50, .50],                                # group 2: adjacent, still short
            [-1.00, .01], [-.99, .02], [-.98, .03],    # global winners for row 2
            [.70, .30],
        ])
        memory.keys.copy_(keys)
        memory._keys_norm.copy_(torch.nn.functional.normalize(keys, dim=-1))
        memory.sample_ids.copy_(torch.arange(12))
        memory._cached_groups = torch.tensor([
            [0, 1, 2], [3, 4, -1], [5, -1, -1],
        ])
        memory._cached_group_sizes = torch.tensor([3, 2, 1])
        memory._cached_extended = torch.tensor([
            [0, 1, 2], [3, 4, 6], [5, 7, -1],
        ])
        memory._cached_extended_sizes = torch.tensor([3, 3, 2])
        queries = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        assignments = torch.tensor([0, 1, 2])

        records = classify_retrieval_branches(memory, 3, 3, assignments)
        _, _, indices = memory.retrieve(queries, 3, hard_assignment=assignments)
        self.assertEqual([r["fallback_type"] for r in records],
                         ["none", "adjacent_region", "global"])
        self.assertEqual(set(indices[0].tolist()), {0, 1, 2})
        self.assertEqual(set(indices[1].tolist()), {3, 4, 6})
        # The extended group has only two candidates. A three-ID return with
        # an outside ID proves the real method selected its global path.
        self.assertTrue(set(indices[2].tolist()) - {5, 7})

    def test_metadata_wrapper_preserves_binary_and_multiclass_outputs(self):
        fixture = ReproductionStateTests()
        fixture.setUp()
        for task in ("binclass", "multiclass"):
            with self.subTest(task=task):
                wrapper, dataset, _ = fixture.make_state(task)
                test_x = dataset._indv_dataset()[2][0]
                _, _, trace, audit = instrument_retrieval(wrapper.model, test_x, batch_size=3)
                self.assertEqual(len(trace), len(test_x))
                self.assertTrue(audit["metadata_off_on_equal"])
                self.assertTrue(audit["model_state_match"])
                self.assertTrue(all(audit["field_equality"].values()))
                self.assertEqual(sum(audit["branch_counts"].values()), len(test_x))


if __name__ == "__main__":
    unittest.main()
