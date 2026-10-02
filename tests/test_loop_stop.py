import unittest

from mlx_lm.loop_stop import LoopStop, LoopStopOptions, parse_loop_stop, repeated_tail

ROW = "<tr>" + "<td></td>" * 12 + "</tr>"


class TestLoopStop(unittest.TestCase):
    def test_runaway_of_empty_rows_is_found(self):
        text = '{"layout": [{"category": "table", "text": "<table>' + ROW * 40
        self.assertEqual(repeated_tail(text, LoopStopOptions()), len(ROW))

    def test_a_blank_form_with_a_few_empty_rows_is_not_a_loop(self):
        text = "<table><tr><td>Date</td><td>Visit</td></tr>" + ROW * 6 + "</table>" + " ".join(f"line {i}" for i in range(120))
        self.assertIsNone(repeated_tail(text, LoopStopOptions()))

    def test_varied_text_is_not_a_loop(self):
        text = " ".join(f"Visit {i} on day {i * 3} with pain {i % 10} of 10." for i in range(200))
        self.assertIsNone(repeated_tail(text, LoopStopOptions()))

    def test_a_short_repeat_must_span_min_chars(self):
        options = LoopStopOptions()
        self.assertIsNone(repeated_tail("a" * 100, options))
        self.assertIsNotNone(repeated_tail("intro " + "-" * 700, options))

    def test_feeding_stops_once_the_loop_is_long_enough(self):
        stop = LoopStop(LoopStopOptions())
        stopped_at = None
        for i in range(200):
            if stop.feed(ROW):
                stopped_at = i + 1
                break
        self.assertIsNotNone(stopped_at)
        self.assertLessEqual(stopped_at, 14)

    def test_parse(self):
        self.assertIsNone(parse_loop_stop(None))
        self.assertIsNone(parse_loop_stop(False))
        self.assertEqual(parse_loop_stop(True).min_repeats, 12)
        self.assertEqual(parse_loop_stop({"min_repeats": 20}).min_repeats, 20)
        with self.assertRaises(ValueError):
            parse_loop_stop({"min_repeats": 0})
        with self.assertRaises(ValueError):
            parse_loop_stop("yes")


if __name__ == "__main__":
    unittest.main()
