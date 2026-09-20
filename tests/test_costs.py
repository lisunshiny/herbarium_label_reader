import unittest
from openrouter_cost import cost_breakdown


class CostTests(unittest.TestCase):
    def test_byok_adds_router_fee_to_separate_upstream_charge(self):
        billing = cost_breakdown({'cost': .01, 'is_byok': True,
                                  'cost_details': {'upstream_inference_cost': .2}})
        self.assertAlmostEqual(billing['total_cost'], .21)
        self.assertEqual(billing['openrouter_cost'], .01)
        self.assertEqual(billing['upstream_inference_cost'], .2)

    def test_regular_router_cost_does_not_double_count_upstream(self):
        for flag in [False, None]:
            self.assertEqual(cost_breakdown({'cost': .2, 'is_byok': flag,
                             'cost_details': {'upstream_inference_cost': .15}})['total_cost'], .2)

    def test_missing_byok_component_is_unknown_not_free(self):
        for usage in [{'cost': 0, 'is_byok': True},
                      {'is_byok': True, 'cost_details': {'upstream_inference_cost': .2}}]:
            self.assertIsNone(cost_breakdown(usage)['total_cost'])
        self.assertEqual(cost_breakdown({'cost': 0, 'is_byok': True,
                         'cost_details': {'upstream_inference_cost': 0}})['total_cost'], 0)
