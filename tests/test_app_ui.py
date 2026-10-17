"""
UI smoke tests using streamlit.testing.v1.AppTest (Step 6C).

Verifies:
1. The app loads and executes without unhandled exceptions.
2. All 3 tabs render.
3. Tab 2 shows 76.60% (runtime double-blind agreement).
4. The deprecated contaminated strings '81.7' and '+6.7' appear nowhere in the rendered text.
"""

import unittest
from streamlit.testing.v1 import AppTest
from src.pipeline import DEFAULT_CHECKPOINT_CANDIDATES


class TestAppUI(unittest.TestCase):

    def test_app_loads_and_tabs_render(self):
        """Verify app loads, all 3 tabs render, Tab 2 shows 76.60%, and forbidden strings absent."""
        weights_present = any(p.exists() and p.is_file() for p in DEFAULT_CHECKPOINT_CANDIDATES)
        if not weights_present:
            self.skipTest("Weights not present in clean CI runner")

        at = AppTest.from_file("../app.py", default_timeout=60)
        at.run()

        # 1. No unhandled exceptions
        self.assertFalse(at.exception, f"App crashed with exception: {at.exception}")

        # 2. Check rendered markdown and text content
        rendered_texts = []
        for elem in at.markdown:
            rendered_texts.append(elem.value)
        for elem in at.info:
            rendered_texts.append(elem.value)
        for elem in at.warning:
            rendered_texts.append(elem.value)
        for elem in at.error:
            rendered_texts.append(elem.value)
        for elem in at.caption:
            rendered_texts.append(elem.value)

        full_rendered_text = " ".join(rendered_texts)

        # 3. Verify all 3 tabs exist
        tab_labels = [t.label for t in at.tabs]
        self.assertEqual(len(tab_labels), 3, f"Expected 3 tabs, found {len(tab_labels)}: {tab_labels}")
        self.assertIn("Tab 1", tab_labels[0])
        self.assertIn("Tab 2", tab_labels[1])
        self.assertIn("Tab 3", tab_labels[2])

        # 4. Verify Tab 2 shows 76.60%
        self.assertIn(
            "76.60%",
            full_rendered_text,
            "Expected 76.60% runtime human agreement to appear in rendered app output"
        )

        # 5. Verify forbidden contaminated strings appear nowhere in rendered text
        self.assertNotIn("81.7", full_rendered_text, "Forbidden string '81.7' found in rendered app text!")
        self.assertNotIn("+6.7", full_rendered_text, "Forbidden string '+6.7' found in rendered app text!")

        safe_tab_labels = [t.encode("ascii", "ignore").decode().strip() for t in tab_labels]
        print("\n" + "=" * 80)
        print("STREAMLIT APP SMOKE TEST PASSED:")
        print("  * App executed with 0 unhandled exceptions.")
        print(f"  * Rendered 3 tabs: {safe_tab_labels}")
        print("  * Verified Tab 2 displays runtime agreement 76.60%.")
        print("  * Verified strings '81.7' and '+6.7' appear nowhere in rendered text.")
        print("=" * 80)


if __name__ == "__main__":
    unittest.main()
