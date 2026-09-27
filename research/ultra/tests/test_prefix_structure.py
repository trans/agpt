import math
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from agpt_ultra.baseline import baseline_epoch, train_baseline
from agpt_ultra.data import CharVocab, make_circular_samples, sorted_circular_samples, sorted_samples
from agpt_ultra.eval import count_objective_tokens, evaluate_sequential_text_loss, evaluate_trie_loss, make_sample_split, split_text
from agpt_ultra.flat_ops import (
    collect_batched_flat_head_evidence,
    compute_flat_hidden_states,
    compute_flat_hidden_states_sequential,
    collect_flat_matrix_free_head_evidence,
    flat_trie_negative_log_likelihood,
    samples_to_flat_trie,
)
from agpt_ultra.fisher import (
    StateEvidence,
    categorical_covariance,
    empirical_distribution,
    empirical_embedding_fisher,
    entropy,
    local_state_evidence,
    local_state_fisher,
    local_state_gradient,
    natural_state_step,
    node_empirical_fisher,
    predicted_distribution,
    reconcile_evidence,
    transition_counts,
)
from agpt_ultra.head_only import (
    batched_head_fisher_matvec,
    collect_matrix_free_head_evidence,
    collect_head_evidence,
    flatten_head,
    head_fisher_matvec,
    head_only_epoch,
    head_shape,
    load_head_theta,
    merge_batched_head_evidence,
    model_head_theta,
    natural_head_step_matrix_free,
    node_head_evidence,
    quadratic_predicted_improvement,
    suggested_quadratic_step_scale,
    train_head_only,
    unflatten_head,
)
from agpt_ultra.hybrid import freeze_embeddings, hybrid_epoch, prefix_hidden_state, train_hybrid
from agpt_ultra.model import TinyCharRNN
from agpt_ultra.objective import (
    assert_state_unchanged,
    immutable_state_dict,
    naive_transition_negative_log_likelihood,
    node_transition_loss,
    trie_negative_log_likelihood,
    trie_node_objectives,
)
from agpt_ultra.prefix_cache import logits_naive, logits_with_prefix_stack, logits_with_prefix_trie
from agpt_ultra.reconcile import (
    GradientModelEvidence,
    LocalModelEvidence,
    apply_gradient_evidence,
    natural_parameter_step,
    quadratic_reconciliation_loss,
    reconcile_child_models,
    reconcile_gradient_evidence,
    reconcile_tree_epoch,
    run_reconciliation_epochs,
)
from agpt_ultra.stats import trie_stats
from agpt_ultra.trie import PrefixTrie, stack_events
from scripts.run_comparison import ComparisonRow, make_model, write_rows
from scripts.profile_unigram_subtries import prefix_range
from scripts.run_prefix_subtree_train import prefix_ranges


class PrefixStructureTest(unittest.TestCase):
    def test_sorted_samples_make_common_prefixes_adjacent(self) -> None:
        samples = sorted_samples("ABCDEFQRS ABCDEFXYZ", block_size=9, stride=1)
        self.assertEqual(samples, sorted(samples))
        self.assertIn("ABCDEFQRS", samples)
        self.assertIn("ABCDEFXYZ", samples)

    def test_circular_samples_wrap_tail_without_extra_start_positions(self) -> None:
        samples = make_circular_samples("ABCDE", block_size=3, stride=1)

        self.assertEqual(samples, ["ABC", "BCD", "CDE", "DEA", "EAB"])

    def test_sorted_circular_samples_can_be_sliced_by_unigram_prefix(self) -> None:
        text = "ABACA"
        vocab = CharVocab.from_text(text)
        samples = sorted_circular_samples(text, block_size=3, stride=1)

        token_index = vocab.stoi["A"]
        lo, hi = prefix_range(samples, vocab.chars, token_index)

        self.assertEqual(samples[lo:hi], ["AAB", "ABA", "ACA"])

    def test_trie_compacts_shared_prefixes(self) -> None:
        samples = ["ABCDEFQRS", "ABCDEFXYZ"]
        trie = PrefixTrie.from_samples(samples)
        raw_nodes_without_sharing = 1 + sum(len(sample) for sample in samples)

        self.assertLess(trie.node_count(), raw_nodes_without_sharing)
        node = trie.root
        for token in "ABCDEF":
            node = node.children[token]
            self.assertEqual(node.prefix_count, 2)
        self.assertEqual(set(node.children), {"Q", "X"})

    def test_stack_events_describe_pop_push_schedule(self) -> None:
        events = stack_events(["ABCDEFQRS", "ABCDEFXYZ"])
        self.assertEqual(events[0].common_prefix, 0)
        self.assertEqual(events[0].push_tokens, tuple("ABCDEFQRS"))
        self.assertEqual(events[1].common_prefix, 6)
        self.assertEqual(events[1].pop_count, 3)
        self.assertEqual(events[1].push_tokens, tuple("XYZ"))

    def test_prefix_stack_logits_match_naive_logits(self) -> None:
        torch.manual_seed(7)
        text = "ABCDEFQRS ABCDEFXYZ"
        vocab = CharVocab.from_text(text)
        samples = ["ABCDEFQRS", "ABCDEFXYZ"]
        model = TinyCharRNN(vocab.size, n_embd=8, n_hidden=16)

        naive = logits_naive(model, vocab, samples)
        cached = logits_with_prefix_stack(model, vocab, samples)

        torch.testing.assert_close(cached, naive)

    def test_prefix_trie_logits_match_naive_logits(self) -> None:
        torch.manual_seed(13)
        text = "ABCDEFQRS ABCDEFXYZ"
        vocab = CharVocab.from_text(text)
        samples = ["ABCDEFQRS", "ABCDEFXYZ"]
        model = TinyCharRNN(vocab.size, n_embd=8, n_hidden=16)

        naive = logits_naive(model, vocab, samples)
        cached = logits_with_prefix_trie(model, vocab, samples)

        torch.testing.assert_close(cached, naive)

    def test_parent_jacobian_distributes_over_child_gradient_sum(self) -> None:
        torch.manual_seed(11)
        parent_jacobian = torch.randn(5, 7)
        child_grads = torch.randn(13, 7)

        left = parent_jacobian @ child_grads.sum(dim=0)
        right = torch.stack([parent_jacobian @ g for g in child_grads]).sum(dim=0)

        torch.testing.assert_close(left, right)

    def test_transition_counts_are_node_count_matrix_row(self) -> None:
        vocab = CharVocab.from_text("ABCD")
        trie = PrefixTrie.from_samples(["AB", "AC", "AC", "AD"])
        counts = transition_counts(trie.root.children["A"], vocab)

        expected = torch.zeros(vocab.size)
        expected[vocab.stoi["B"]] = 1
        expected[vocab.stoi["C"]] = 2
        expected[vocab.stoi["D"]] = 1

        torch.testing.assert_close(counts, expected)

    def test_categorical_covariance_matches_closed_form(self) -> None:
        q = torch.tensor([0.25, 0.5, 0.25])
        cov = categorical_covariance(q)

        torch.testing.assert_close(cov, torch.diag(q) - torch.outer(q, q))
        torch.testing.assert_close(cov.sum(dim=0), torch.zeros(3))
        torch.testing.assert_close(cov.sum(dim=1), torch.zeros(3))

    def test_node_empirical_fisher_matches_w_cov_w(self) -> None:
        torch.manual_seed(17)
        vocab = CharVocab.from_text("ABCD")
        trie = PrefixTrie.from_samples(["AB", "AC", "AC", "AD"])
        node = trie.root.children["A"]
        output_weight = torch.randn(vocab.size, 5)

        counts = transition_counts(node, vocab)
        q = empirical_distribution(counts)
        expected = output_weight.T @ (torch.diag(q) - torch.outer(q, q)) @ output_weight

        torch.testing.assert_close(node_empirical_fisher(node, vocab, output_weight), expected)

    def test_fisher_is_derived_from_aggregate_counts_not_raw_covariance_sum(self) -> None:
        counts_a = torch.tensor([3.0, 0.0, 0.0])
        counts_b = torch.tensor([0.0, 1.0, 2.0])
        output_weight = torch.eye(3)

        aggregate = empirical_embedding_fisher(counts_a + counts_b, output_weight)
        weighted_child_sum = (
            counts_a.sum() * empirical_embedding_fisher(counts_a, output_weight)
            + counts_b.sum() * empirical_embedding_fisher(counts_b, output_weight)
        ) / (counts_a.sum() + counts_b.sum())

        self.assertGreater((aggregate - weighted_child_sum).abs().max().item(), 1e-6)

    def test_entropy_summarizes_logit_fisher_spectrum_trace(self) -> None:
        q = torch.tensor([0.5, 0.25, 0.25])
        cov = categorical_covariance(q)

        self.assertGreater(entropy(q).item(), 0.0)
        torch.testing.assert_close(torch.trace(cov), 1.0 - (q * q).sum())

    def test_local_state_gradient_uses_predicted_minus_target(self) -> None:
        torch.manual_seed(19)
        output_weight = torch.randn(4, 3)
        hidden = torch.randn(3)
        target = torch.tensor([0.1, 0.2, 0.3, 0.4])

        predicted = predicted_distribution(hidden, output_weight)
        gradient = local_state_gradient(predicted, target, output_weight)

        torch.testing.assert_close(gradient, output_weight.T @ (predicted - target))

    def test_local_state_fisher_uses_predicted_distribution(self) -> None:
        torch.manual_seed(23)
        output_weight = torch.randn(4, 3)
        hidden = torch.randn(3)
        target = torch.tensor([0.1, 0.2, 0.3, 0.4])

        evidence = local_state_evidence(hidden, target, output_weight)
        predicted = predicted_distribution(hidden, output_weight)
        expected_fisher = output_weight.T @ categorical_covariance(predicted) @ output_weight

        torch.testing.assert_close(evidence.fisher, expected_fisher)
        torch.testing.assert_close(evidence.fisher, local_state_fisher(predicted, output_weight))

    def test_natural_state_step_solves_damped_fisher_system(self) -> None:
        gradient = torch.tensor([1.0, -2.0])
        fisher = torch.tensor([[3.0, 0.5], [0.5, 2.0]])
        evidence = StateEvidence(gradient=gradient, fisher=fisher)
        damping = 0.25

        step = natural_state_step(evidence, damping=damping)
        expected = -torch.linalg.solve(fisher + damping * torch.eye(2), gradient)

        torch.testing.assert_close(step, expected)

    def test_parent_reconciles_local_and_child_evidence_by_summing(self) -> None:
        local = StateEvidence(
            gradient=torch.tensor([1.0, 2.0]),
            fisher=torch.tensor([[2.0, 0.1], [0.1, 3.0]]),
        )
        children = [
            StateEvidence(
                gradient=torch.tensor([0.5, -1.0]),
                fisher=torch.tensor([[1.0, 0.2], [0.2, 1.5]]),
            ),
            StateEvidence(
                gradient=torch.tensor([-0.25, 0.75]),
                fisher=torch.tensor([[0.5, 0.0], [0.0, 0.25]]),
            ),
        ]

        reconciled = reconcile_evidence(local, children)

        torch.testing.assert_close(reconciled.gradient, torch.tensor([1.25, 1.75]))
        torch.testing.assert_close(reconciled.fisher, torch.tensor([[3.5, 0.3], [0.3, 4.75]]))

    def test_node_transition_loss_matches_closed_form_objective_term(self) -> None:
        logits = torch.tensor([0.2, -0.3, 0.8])
        counts = torch.tensor([2.0, 0.0, 3.0])

        expected = -(counts * torch.log_softmax(logits, dim=0)).sum()

        torch.testing.assert_close(node_transition_loss(logits, counts), expected)

    def test_trie_objective_matches_naive_next_token_loss(self) -> None:
        torch.manual_seed(29)
        text = "ABCDEFQRS ABCDEFXYZ"
        vocab = CharVocab.from_text(text)
        samples = ["ABCDEFQRS", "ABCDEFXYZ"]
        model = TinyCharRNN(vocab.size, n_embd=8, n_hidden=16)

        trie_loss = trie_negative_log_likelihood(model, vocab, samples)
        naive_loss = naive_transition_negative_log_likelihood(model, vocab, samples)

        torch.testing.assert_close(trie_loss, naive_loss)

    def test_trie_node_objectives_hold_prefix_counts_and_hidden_states(self) -> None:
        torch.manual_seed(31)
        vocab = CharVocab.from_text("ABC")
        samples = ["AB", "AC", "AC"]
        model = TinyCharRNN(vocab.size, n_embd=8, n_hidden=16)

        objectives = {objective.prefix: objective for objective in trie_node_objectives(model, vocab, samples)}

        self.assertEqual(set(objectives), {"", "A"})
        self.assertEqual(objectives[""].count, 3)
        self.assertEqual(objectives["A"].count, 3)
        self.assertEqual(objectives["A"].counts[vocab.stoi["B"]].item(), 1.0)
        self.assertEqual(objectives["A"].counts[vocab.stoi["C"]].item(), 2.0)
        self.assertEqual(tuple(objectives["A"].hidden.shape), (16,))

    def test_local_parameter_snapshot_is_immutable_to_local_update(self) -> None:
        torch.manual_seed(37)
        model = TinyCharRNN(vocab_size=4, n_embd=8, n_hidden=16)
        snapshot = immutable_state_dict(model)
        local_head_weight = snapshot["head.weight"].clone()

        local_head_weight.add_(1.0)

        assert_state_unchanged(snapshot, model)
        self.assertGreater((local_head_weight - model.head.weight).abs().max().item(), 0.0)

    def test_child_models_reconcile_by_fisher_weighted_geometry(self) -> None:
        children = [
            LocalModelEvidence(
                theta=torch.tensor([1.0, 0.0]),
                fisher=torch.tensor([[10.0, 0.0], [0.0, 1.0]]),
            ),
            LocalModelEvidence(
                theta=torch.tensor([0.0, 2.0]),
                fisher=torch.tensor([[1.0, 0.0], [0.0, 10.0]]),
            ),
        ]

        parent = reconcile_child_models(children, damping=0.0)

        torch.testing.assert_close(parent, torch.tensor([10.0 / 11.0, 20.0 / 11.0]))
        self.assertFalse(torch.allclose(parent, torch.stack([child.theta for child in children]).mean(dim=0)))

    def test_reconciled_model_minimizes_child_quadratic_objective(self) -> None:
        children = [
            LocalModelEvidence(
                theta=torch.tensor([1.0, -1.0]),
                fisher=torch.tensor([[3.0, 0.5], [0.5, 2.0]]),
            ),
            LocalModelEvidence(
                theta=torch.tensor([-0.5, 2.0]),
                fisher=torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
            ),
        ]
        parent = reconcile_child_models(children, damping=0.0)
        parent_loss = quadratic_reconciliation_loss(parent, children)
        nearby_loss = quadratic_reconciliation_loss(parent + torch.tensor([0.1, -0.1]), children)

        self.assertLess(parent_loss.item(), nearby_loss.item())

    def test_tree_epoch_broadcasts_global_theta_to_each_node_and_returns_root_model(self) -> None:
        trie = PrefixTrie.from_samples(["AB", "AC"])
        initial_theta = torch.tensor([0.0, 0.0])
        seen_base_thetas: dict[str, torch.Tensor] = {}

        def local_update(prefix: str, node, base_theta: torch.Tensor) -> LocalModelEvidence:
            seen_base_thetas[prefix] = base_theta.clone()
            offset = torch.tensor([float(node.depth), float(node.count)])
            return LocalModelEvidence(theta=base_theta + offset, fisher=torch.eye(2))

        result = reconcile_tree_epoch(trie.root, initial_theta, local_update, damping=0.0)

        self.assertEqual(set(seen_base_thetas), {"", "A", "AB", "AC"})
        for base_theta in seen_base_thetas.values():
            torch.testing.assert_close(base_theta, initial_theta)
        torch.testing.assert_close(result.global_theta, result.node_evidence[""].theta)
        self.assertEqual(result.root_evidence.fisher.shape, (2, 2))

    def test_reconciliation_epochs_feed_root_model_into_next_epoch(self) -> None:
        trie = PrefixTrie.from_samples(["AB"])
        initial_theta = torch.tensor([0.0])
        epoch_root_inputs: list[torch.Tensor] = []

        def local_update(epoch: int, prefix: str, node, base_theta: torch.Tensor) -> LocalModelEvidence:
            if prefix == "":
                epoch_root_inputs.append(base_theta.clone())
            proposal = base_theta + torch.tensor([1.0 + epoch])
            return LocalModelEvidence(theta=proposal, fisher=torch.eye(1))

        results = run_reconciliation_epochs(
            trie.root,
            initial_theta,
            local_update,
            epochs=3,
            damping=0.0,
        )

        torch.testing.assert_close(epoch_root_inputs[0], initial_theta)
        torch.testing.assert_close(epoch_root_inputs[1], results[0].global_theta)
        torch.testing.assert_close(epoch_root_inputs[2], results[1].global_theta)
        self.assertGreater(results[-1].global_theta.item(), initial_theta.item())

    def test_gradient_evidence_sums_before_natural_parameter_step(self) -> None:
        children = [
            GradientModelEvidence(
                gradient=torch.tensor([1.0, -2.0]),
                fisher=torch.tensor([[3.0, 0.5], [0.5, 2.0]]),
            ),
            GradientModelEvidence(
                gradient=torch.tensor([-0.25, 0.5]),
                fisher=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            ),
        ]

        evidence = reconcile_gradient_evidence(children)
        step = natural_parameter_step(evidence, damping=0.1)
        expected_gradient = torch.tensor([0.75, -1.5])
        expected_fisher = torch.tensor([[4.0, 0.5], [0.5, 6.0]])
        expected_step = -torch.linalg.solve(expected_fisher + 0.1 * torch.eye(2), expected_gradient)

        torch.testing.assert_close(evidence.gradient, expected_gradient)
        torch.testing.assert_close(evidence.fisher, expected_fisher)
        torch.testing.assert_close(step, expected_step)

    def test_gradient_update_matches_fisher_weighted_theta_reconciliation_without_damping(self) -> None:
        theta0 = torch.tensor([0.5, -0.5])
        children_g = [
            GradientModelEvidence(
                gradient=torch.tensor([1.0, 0.0]),
                fisher=torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
            ),
            GradientModelEvidence(
                gradient=torch.tensor([0.0, -3.0]),
                fisher=torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
            ),
        ]
        children_theta = [
            LocalModelEvidence(
                theta=theta0 + natural_parameter_step(child, damping=0.0),
                fisher=child.fisher,
            )
            for child in children_g
        ]

        from_gradients = apply_gradient_evidence(theta0, children_g, damping=0.0)
        from_thetas = reconcile_child_models(children_theta, damping=0.0)

        torch.testing.assert_close(from_gradients, from_thetas)

    def test_head_theta_round_trips_through_flat_vector(self) -> None:
        torch.manual_seed(41)
        model = TinyCharRNN(vocab_size=5, n_embd=4, n_hidden=3)
        theta = model_head_theta(model)
        weight, bias = unflatten_head(theta, head_shape(model))

        reflattened = flatten_head(weight, bias)
        load_head_theta(model, reflattened)

        torch.testing.assert_close(theta, reflattened)
        torch.testing.assert_close(model.head.weight, weight)
        torch.testing.assert_close(model.head.bias, bias)

    def test_head_only_node_gradient_matches_autograd(self) -> None:
        torch.manual_seed(43)
        vocab_size = 4
        hidden_size = 3
        weight = torch.randn(vocab_size, hidden_size)
        bias = torch.randn(vocab_size)
        theta = flatten_head(weight, bias).detach().clone().requires_grad_(True)
        shape = head_shape(TinyCharRNN(vocab_size=vocab_size, n_embd=4, n_hidden=hidden_size))
        hidden = torch.randn(hidden_size)
        counts = torch.tensor([2.0, 0.0, 1.0, 3.0])

        evidence = node_head_evidence(theta.detach(), shape, hidden, counts)
        logits = (unflatten_head(theta, shape)[0] @ hidden) + unflatten_head(theta, shape)[1]
        loss = -(counts * torch.log_softmax(logits, dim=0)).sum()
        loss.backward()

        torch.testing.assert_close(evidence.gradient, theta.grad)
        self.assertEqual(evidence.fisher.shape, (theta.numel(), theta.numel()))

    def test_collect_head_evidence_matches_autograd_for_trie_loss(self) -> None:
        torch.manual_seed(47)
        text = "ABACABADBAE"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)

        evidence = collect_head_evidence(model, vocab, samples)
        loss = trie_negative_log_likelihood(model, vocab, samples)
        loss.backward()
        autograd_gradient = flatten_head(model.head.weight.grad, model.head.bias.grad)

        torch.testing.assert_close(evidence.gradient, autograd_gradient, atol=1e-5, rtol=1e-5)
        self.assertEqual(evidence.fisher.shape, (evidence.gradient.numel(), evidence.gradient.numel()))

    def test_matrix_free_head_evidence_matches_dense_gradient_and_matvec(self) -> None:
        torch.manual_seed(73)
        text = "ABACABAD"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        theta = model_head_theta(model)

        dense = collect_head_evidence(model, vocab, samples, theta)
        matrix_free = collect_matrix_free_head_evidence(model, vocab, samples, theta)
        vector = torch.randn_like(theta)

        torch.testing.assert_close(matrix_free.gradient, dense.gradient, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            head_fisher_matvec(vector, matrix_free),
            dense.fisher @ vector,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_matrix_free_cg_matches_dense_natural_head_step(self) -> None:
        torch.manual_seed(79)
        text = "ABACABAD"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        theta = model_head_theta(model)
        damping = 0.75

        dense = collect_head_evidence(model, vocab, samples, theta)
        matrix_free = collect_matrix_free_head_evidence(model, vocab, samples, theta)
        cg = natural_head_step_matrix_free(
            matrix_free,
            damping=damping,
            max_cg_iter=theta.numel(),
            cg_tolerance=1e-8,
        )
        dense_step = -torch.linalg.solve(dense.fisher + damping * torch.eye(theta.numel()), dense.gradient)

        torch.testing.assert_close(cg.solution, dense_step, atol=1e-4, rtol=1e-4)

    def test_suggested_quadratic_step_scale_uses_fisher_quadratic(self) -> None:
        gradient = torch.tensor([2.0, -1.0])
        delta = torch.tensor([-0.5, 0.25])
        fisher_delta = torch.tensor([-1.0, 0.5])

        scale = suggested_quadratic_step_scale(gradient, delta, fisher_delta, max_step_scale=10.0)
        expected = -gradient.dot(delta).item() / delta.dot(fisher_delta).item()

        self.assertAlmostEqual(scale, expected)

    def test_quadratic_predicted_improvement_uses_selected_scale(self) -> None:
        gradient = torch.tensor([2.0, -1.0])
        delta = torch.tensor([-0.5, 0.25])
        fisher_delta = torch.tensor([-1.0, 0.5])

        improvement = quadratic_predicted_improvement(gradient, delta, fisher_delta, step_scale=0.5)
        expected = -(0.5 * gradient.dot(delta).item() + 0.5 * 0.5 * 0.5 * delta.dot(fisher_delta).item())

        self.assertAlmostEqual(improvement, expected)

    def test_head_only_auto_step_with_line_search_reduces_loss(self) -> None:
        torch.manual_seed(113)
        text = "ABACABADABACABAD"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=4, stride=2)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        flat = samples_to_flat_trie(samples, vocab)

        result = head_only_epoch(
            model,
            vocab,
            samples,
            damping=3.0,
            step_scale="auto",
            max_step_scale=1.0,
            line_search_steps=4,
            flat_trie=flat,
            max_cg_iter=8,
        )

        self.assertIsNotNone(result.suggested_step_scale)
        self.assertIsNotNone(result.step_scale)
        self.assertIsNotNone(result.predicted_improvement)
        self.assertIsNotNone(result.actual_improvement)
        self.assertIsNotNone(result.improvement_ratio)
        self.assertLess(result.after_loss, result.before_loss)

    def test_flat_trie_loss_matches_object_trie_loss(self) -> None:
        torch.manual_seed(103)
        text = "ABACABAD"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        flat = samples_to_flat_trie(samples, vocab)

        torch.testing.assert_close(
            flat_trie_negative_log_likelihood(model, flat),
            trie_negative_log_likelihood(model, vocab, samples),
        )

    def test_flat_trie_loss_can_start_from_prefix_hidden_state(self) -> None:
        torch.manual_seed(104)
        text = "ABCD"
        vocab = CharVocab.from_text(text)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=5)
        suffix_trie = samples_to_flat_trie(["BC"], vocab)
        initial_hidden = prefix_hidden_state(model, vocab, "A")

        loss = flat_trie_negative_log_likelihood(model, suffix_trie, initial_hidden=initial_hidden)

        with torch.no_grad():
            state_a = prefix_hidden_state(model, vocab, "A").unsqueeze(0)
            logits_b = model.head(state_a)
            _, state_ab = model.step(torch.tensor([vocab.stoi["B"]]), state_a)
            logits_c = model.head(state_ab)
            expected = torch.nn.functional.cross_entropy(logits_b, torch.tensor([vocab.stoi["B"]]), reduction="sum")
            expected = expected + torch.nn.functional.cross_entropy(logits_c, torch.tensor([vocab.stoi["C"]]), reduction="sum")

        torch.testing.assert_close(loss, expected)

    def test_depth_batched_flat_hidden_states_match_sequential(self) -> None:
        torch.manual_seed(104)
        text = "ABACABADABAE"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA", "BAE"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        flat = samples_to_flat_trie(samples, vocab)

        sequential = compute_flat_hidden_states_sequential(model, flat)
        batched = compute_flat_hidden_states(model, flat)

        torch.testing.assert_close(batched, sequential)

    def test_flat_head_evidence_matches_object_head_evidence(self) -> None:
        torch.manual_seed(107)
        text = "ABACABAD"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        theta = model_head_theta(model)
        flat = samples_to_flat_trie(samples, vocab)
        dense = collect_head_evidence(model, vocab, samples, theta)
        flat_evidence = collect_flat_matrix_free_head_evidence(model, flat, theta)
        vector = torch.randn_like(theta)

        torch.testing.assert_close(flat_evidence.gradient, dense.gradient, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            head_fisher_matvec(vector, flat_evidence),
            dense.fisher @ vector,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_batched_flat_head_evidence_matches_dense_head_evidence(self) -> None:
        torch.manual_seed(109)
        text = "ABACABAD"
        vocab = CharVocab.from_text(text)
        samples = ["ABA", "ACA", "ADA"]
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        theta = model_head_theta(model)
        flat = samples_to_flat_trie(samples, vocab)
        dense = collect_head_evidence(model, vocab, samples, theta)
        batched = collect_batched_flat_head_evidence(model, flat, theta)
        vector = torch.randn_like(theta)

        torch.testing.assert_close(batched.gradient, dense.gradient, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            batched_head_fisher_matvec(vector, batched),
            dense.fisher @ vector,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_prefix_subtree_head_evidence_merges_to_whole_trie_evidence(self) -> None:
        torch.manual_seed(110)
        text = "ABACABADBAE"
        vocab = CharVocab.from_text(text)
        samples = sorted(["ABA", "ACA", "BAE", "BAD"])
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        theta = model_head_theta(model)
        whole = collect_batched_flat_head_evidence(model, samples_to_flat_trie(samples, vocab), theta)
        root = collect_batched_flat_head_evidence(
            model,
            samples_to_flat_trie([sample[:1] for sample in samples], vocab),
            theta,
        )
        pieces = [root]
        for prefix, start, end in prefix_ranges(samples, 1):
            suffixes = [sample[1:] for sample in samples[start:end]]
            subtree = samples_to_flat_trie(suffixes, vocab)
            pieces.append(
                collect_batched_flat_head_evidence(
                    model,
                    subtree,
                    theta,
                    initial_hidden=prefix_hidden_state(model, vocab, prefix),
                )
            )
        merged = merge_batched_head_evidence(pieces)
        vector = torch.randn_like(theta)

        torch.testing.assert_close(merged.gradient, whole.gradient, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            batched_head_fisher_matvec(vector, merged),
            batched_head_fisher_matvec(vector, whole),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_prefix_subtree_body_gradients_match_whole_trie_gradients(self) -> None:
        torch.manual_seed(111)
        text = "ABACABADBAE"
        vocab = CharVocab.from_text(text)
        samples = sorted(["ABA", "ACA", "BAE", "BAD"])
        whole_model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        split_model = deepcopy(whole_model)
        for model in (whole_model, split_model):
            for param in model.head.parameters():
                param.requires_grad_(False)

        flat_trie_negative_log_likelihood(whole_model, samples_to_flat_trie(samples, vocab)).backward()
        whole_grads = [param.grad.detach().clone() for param in whole_model.cell.parameters()]

        top = samples_to_flat_trie([sample[:1] for sample in samples], vocab)
        flat_trie_negative_log_likelihood(split_model, top).backward()
        for prefix, start, end in prefix_ranges(samples, 1):
            suffixes = [sample[1:] for sample in samples[start:end]]
            subtree = samples_to_flat_trie(suffixes, vocab)
            flat_trie_negative_log_likelihood(
                split_model,
                subtree,
                initial_hidden=prefix_hidden_state(split_model, vocab, prefix),
            ).backward()
        split_grads = [param.grad.detach().clone() for param in split_model.cell.parameters()]

        for split_grad, whole_grad in zip(split_grads, whole_grads, strict=True):
            torch.testing.assert_close(split_grad, whole_grad, atol=1e-5, rtol=1e-5)

    def test_head_only_epoch_reduces_trie_loss(self) -> None:
        torch.manual_seed(53)
        text = "First Citizen:\nSpeak, speak.\nFirst Citizen:\nSpeak again.\n"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=8, stride=4)
        model = TinyCharRNN(vocab.size, n_embd=8, n_hidden=6)

        result = head_only_epoch(model, vocab, samples, damping=5.0, step_scale=0.5)

        self.assertLess(result.after_loss, result.before_loss)

    def test_head_only_training_feeds_updated_head_to_next_epoch(self) -> None:
        torch.manual_seed(59)
        text = "ABACABADABACABAD"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=4, stride=2)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)

        results = train_head_only(model, vocab, samples, epochs=2, damping=3.0, step_scale=0.5)

        self.assertEqual(len(results), 2)
        torch.testing.assert_close(model_head_theta(model), results[-1].theta)
        self.assertLess(results[-1].after_loss, results[0].before_loss)

    def test_hybrid_epoch_updates_cell_and_head_but_not_embedding(self) -> None:
        torch.manual_seed(61)
        text = "ABACABADABACABAD"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=4, stride=2)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        optimizer = torch.optim.AdamW(model.cell.parameters(), lr=1e-2)

        before_embed = model.embed.weight.detach().clone()
        before_cell = {name: param.detach().clone() for name, param in model.cell.named_parameters()}
        before_head = model_head_theta(model)

        result = hybrid_epoch(
            model,
            vocab,
            samples,
            body_optimizer=optimizer,
            head_damping=3.0,
            head_step_scale=0.5,
            max_grad_norm=1.0,
        )

        torch.testing.assert_close(model.embed.weight, before_embed)
        self.assertTrue(
            any(
                not torch.allclose(param, before_cell[name])
                for name, param in model.cell.named_parameters()
            )
        )
        self.assertGreater((model_head_theta(model) - before_head).abs().max().item(), 0.0)
        self.assertLess(result.after_loss, result.before_loss)

    def test_hybrid_epoch_can_update_embeddings(self) -> None:
        torch.manual_seed(63)
        text = "ABACABADABACABAD"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=4, stride=2)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        optimizer = torch.optim.AdamW([*model.cell.parameters(), *model.embed.parameters()], lr=1e-2)

        before_embed = model.embed.weight.detach().clone()

        hybrid_epoch(
            model,
            vocab,
            samples,
            body_optimizer=optimizer,
            head_damping=3.0,
            head_step_scale=0.5,
            max_grad_norm=1.0,
            update_embeddings=True,
        )

        self.assertGreater((model.embed.weight - before_embed).abs().max().item(), 0.0)

    def test_hybrid_training_reduces_loss_across_epochs(self) -> None:
        torch.manual_seed(67)
        text = "First Citizen:\nSpeak, speak.\nFirst Citizen:\nSpeak again.\n"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=8, stride=4)
        model = TinyCharRNN(vocab.size, n_embd=6, n_hidden=5)

        freeze_embeddings(model)
        before_embed = model.embed.weight.detach().clone()
        results = train_hybrid(
            model,
            vocab,
            samples,
            epochs=3,
            body_lr=5e-3,
            head_damping=5.0,
            head_step_scale=0.5,
            max_grad_norm=1.0,
        )

        torch.testing.assert_close(model.embed.weight, before_embed)
        self.assertLess(results[-1].after_loss, results[0].before_loss)

    def test_baseline_epoch_updates_cell_and_head_but_not_embedding(self) -> None:
        torch.manual_seed(83)
        text = "ABACABADABACABAD"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=4, stride=2)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)
        optimizer = torch.optim.AdamW([*model.cell.parameters(), *model.head.parameters()], lr=1e-2)

        before_embed = model.embed.weight.detach().clone()
        before_cell = {name: param.detach().clone() for name, param in model.cell.named_parameters()}
        before_head = model_head_theta(model)

        result = baseline_epoch(model, vocab, samples, optimizer=optimizer, max_grad_norm=1.0)

        torch.testing.assert_close(model.embed.weight, before_embed)
        self.assertTrue(
            any(
                not torch.allclose(param, before_cell[name])
                for name, param in model.cell.named_parameters()
            )
        )
        self.assertGreater((model_head_theta(model) - before_head).abs().max().item(), 0.0)
        self.assertLess(result.after_loss, result.before_loss)

    def test_baseline_training_reduces_loss_across_epochs(self) -> None:
        torch.manual_seed(89)
        text = "First Citizen:\nSpeak, speak.\nFirst Citizen:\nSpeak again.\n"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=8, stride=4)
        model = TinyCharRNN(vocab.size, n_embd=6, n_hidden=5)

        before_embed = model.embed.weight.detach().clone()
        results = train_baseline(
            model,
            vocab,
            samples,
            epochs=3,
            lr=5e-3,
            max_grad_norm=1.0,
        )

        torch.testing.assert_close(model.embed.weight, before_embed)
        self.assertLess(results[-1].after_loss, results[0].before_loss)

    def test_comparison_model_initialization_is_seed_stable(self) -> None:
        model_a = make_model(seed=101, vocab_size=5, embedding_size=4, hidden_size=3)
        model_b = make_model(seed=101, vocab_size=5, embedding_size=4, hidden_size=3)

        for param_a, param_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
            torch.testing.assert_close(param_a, param_b)
        self.assertFalse(model_a.embed.weight.requires_grad)

    def test_comparison_rows_write_csv(self) -> None:
        rows = [
            ComparisonRow(
                seed=1,
                method="baseline",
                epoch=0,
                train_loss=float("nan"),
                val_nll=4.0,
                val_ppl=55.0,
                runtime_sec=0.0,
                cg_iters=None,
                step_scale=None,
                suggested_step_scale=None,
                predicted_improvement=None,
                actual_improvement=None,
                improvement_ratio=None,
            ),
            ComparisonRow(
                seed=1,
                method="hybrid",
                epoch=1,
                train_loss=10.0,
                val_nll=3.0,
                val_ppl=20.0,
                runtime_sec=1.5,
                cg_iters=8,
                step_scale=0.5,
                suggested_step_scale=0.75,
                predicted_improvement=2.0,
                actual_improvement=1.0,
                improvement_ratio=0.5,
            ),
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.csv"
            write_rows(path, rows)
            text = path.read_text(encoding="utf-8")

        self.assertIn("seed,method,epoch,train_loss,val_nll,val_ppl,runtime_sec,cg_iters,step_scale,suggested_step_scale,predicted_improvement,actual_improvement,improvement_ratio", text)
        self.assertIn("baseline", text)
        self.assertIn("hybrid", text)

    def test_trie_stats_measure_singletons_and_branching(self) -> None:
        vocab = CharVocab.from_text("ABCD")
        samples = ["AB", "AC", "AD"]
        stats = trie_stats(samples, vocab)

        self.assertEqual(stats.samples, 3)
        self.assertEqual(stats.block_size, 2)
        self.assertEqual(stats.max_depth, 2)
        self.assertEqual(stats.predictive_nodes, 2)
        self.assertEqual(stats.branching_nodes, 1)
        self.assertEqual(stats.singleton_nodes, 0)
        self.assertAlmostEqual(stats.branching_fraction, 0.5)
        self.assertGreater(stats.weighted_mean_entropy, 0.0)

    def test_make_sample_split_keeps_train_and_validation_separate(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        split = split_text(text, train_fraction=0.5)
        samples = make_sample_split(
            text,
            block_size=4,
            stride=2,
            train_fraction=0.5,
            max_train_samples=2,
            max_val_samples=3,
        )

        self.assertEqual(split.train_text, "abcdefghijklm")
        self.assertEqual(split.val_text, "nopqrstuvwxyz")
        self.assertTrue(all(sample in split.train_text for sample in samples.train_samples))
        self.assertTrue(all(sample in split.val_text for sample in samples.val_samples))
        self.assertEqual(len(samples.train_samples), 2)
        self.assertEqual(len(samples.val_samples), 3)

    def test_evaluate_trie_loss_reports_per_token_metrics(self) -> None:
        torch.manual_seed(71)
        text = "ABACABADABACABAD"
        vocab = CharVocab.from_text(text)
        samples = sorted_samples(text, block_size=4, stride=2)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)

        metrics = evaluate_trie_loss(model, vocab, samples)

        self.assertEqual(metrics.tokens, count_objective_tokens(samples))
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreater(metrics.nll_per_token, 0.0)
        self.assertGreater(metrics.perplexity, 1.0)

    def test_evaluate_sequential_text_loss_matches_manual_recurrent_pass(self) -> None:
        torch.manual_seed(73)
        text = "ABACABAD"
        vocab = CharVocab.from_text(text)
        model = TinyCharRNN(vocab.size, n_embd=4, n_hidden=3)

        metrics = evaluate_sequential_text_loss(model, vocab, text, chunk_size=3)

        ids = torch.tensor(vocab.encode(text), dtype=torch.long)
        state = model.initial_state(1)
        expected_loss = torch.tensor(0.0)
        with torch.no_grad():
            for position in range(ids.numel() - 1):
                logits, state = model.step(ids[position : position + 1], state)
                expected_loss = expected_loss + torch.nn.functional.cross_entropy(
                    logits,
                    ids[position + 1 : position + 2],
                    reduction="sum",
                )
        expected_nll = expected_loss.item() / (len(text) - 1)

        self.assertEqual(metrics.tokens, len(text) - 1)
        self.assertAlmostEqual(metrics.loss, expected_loss.item(), places=6)
        self.assertAlmostEqual(metrics.nll_per_token, expected_nll, places=6)
        self.assertAlmostEqual(metrics.bits_per_char, expected_nll / math.log(2.0), places=6)


if __name__ == "__main__":
    unittest.main()
