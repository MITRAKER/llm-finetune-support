"""Unit tests for the automatic evaluation metrics."""

from __future__ import annotations

import pytest

import metrics as m


def test_strip_reasoning():
    assert m.strip_reasoning("<think>\nhmm\n</think>\n\nHello!") == "Hello!"
    assert m.strip_reasoning("Hi <think>cut off mid-way") == "Hi"
    assert m.strip_reasoning("No reasoning here.") == "No reasoning here."


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Call us at 1-800-555-0199 any time.", "phone"),
        ("Call us at +44 20 7946 0958.", "phone"),
        ("Email help@shop-example.com for help.", "email"),
        ("Visit www.example-store.com/returns", "url"),
        ("Go to https://store.test/orders", "url"),
        ("Your tracking number is 9400111899223.", "long_number"),
        ("Your order #482913 has shipped.", "long_number"),
        ("You will be refunded $49.99.", "money"),
        ("A fee of 15 dollars applies.", "money"),
        ("Call us at (555)123-4567.", "phone"),
        ("Call us at 555-1234.", "phone"),
        ("Call +1 555 0100 for help.", "phone"),
        ("Your tracking number is 1Z999AA10123456784.", "code"),
        ("Use the discount code SAVE100 at checkout.", "code"),
        ("Visit shop.example.de for details.", "url"),
    ],
)
def test_invented_details_flags_made_up_specifics(text, kind):
    assert kind in m.invented_details(text)


@pytest.mark.parametrize(
    "text",
    [
        "It was placed on 12.05.2024 and shipped on 14.05.2024.",
        "It was placed on 12-05-2024.",
        "Delivery takes 3-5 business days; chat is open 9:00-17:00, 24/7.",
        "COVID-19 rules may delay some parcels.",
    ],
)
def test_dates_and_ordinary_numbers_are_not_flagged(text):
    assert m.invented_details(text) == {}


@pytest.mark.parametrize("reply", ["You will be refunded $88.", "A refund of $8,823 was issued.", "You get $231 back."])
def test_numbers_inside_the_order_number_do_not_excuse_an_invented_amount(reply):
    """The customer's order number 88231 must not excuse a made-up refund of $88 (exact match, not substring)."""
    found = m.invented_details(reply, user_message="How much exactly will I be refunded for order 88231?")
    assert "money" in found


def test_details_the_customer_gave_may_be_repeated_in_any_format():
    assert m.invented_details("Order #88231 has shipped.", user_message="Where is order 88231?") == {}
    assert m.invented_details("I see the charge of $49.99.", user_message="I was charged 49.99 twice") == {}
    assert m.invented_details("Code SAVE100 is applied.", user_message="my code save100 fails") == {}
    assert m.invented_details("Refunds over $10,000 need approval.", user_message="Refund over $10,000?") == {}
    assert "money" in m.invented_details("We can refund $100 now.", user_message="Refund over $10,000?")


def test_placeholders_and_user_supplied_details_are_not_flagged():
    text = "Please call {{Customer Support Phone Number}} or visit {{Website URL}} about order 88231."
    assert m.invented_details(text, user_message="Where is order 88231?") == {}
    assert m.invented_details("I can help you cancel your order.") == {}


def test_contiguous_digits_count_as_long_number_not_phone():
    found = m.invented_details("Call 18005550199 now")
    assert "long_number" in found
    assert "phone" not in found


def test_dates_and_versions_are_not_phone_numbers():
    assert "phone" not in m.invented_details("Delivered on 2024-05-12 by app version 1.2.3")


def test_uses_placeholder():
    assert m.uses_placeholder("Your order {{Order Number}} is on its way")
    assert not m.uses_placeholder("Your order is on its way")


def test_rouge_l_identical_and_disjoint():
    scores = m.rouge_l(["the order was cancelled", "zebra"], ["the order was cancelled", "refund issued today"])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_intent_classifier_and_summary():
    train_texts = [
        "I have cancelled your order {{Order Number}} as requested",
        "Your order cancellation is confirmed",
        "We can cancel the purchase for you",
        "cancel order steps are easy",
        "Your refund has been issued to your card",
        "Refunds take five days to appear",
        "I will process the refund for you",
        "refund request received",
    ] * 3
    train_intents = (["cancel_order"] * 4 + ["get_refund"] * 4) * 3
    clf = m.IntentClassifier().fit(train_texts, train_intents)
    assert clf.predict(["your refund was issued"]) == ["get_refund"]
    assert clf.accuracy(["we cancelled the order", "refund issued"], ["cancel_order", "get_refund"]) == 1.0

    summary, rows = m.summarize(
        predictions=["Your refund has been issued, call 1-800-555-0199", "I cancelled order {{Order Number}}"],
        references=["Your refund has been issued", "I cancelled your order {{Order Number}}"],
        user_messages=["refund please", "cancel my order"],
        true_intents=["get_refund", "cancel_order"],
        classifier=clf,
    )
    assert summary.n == 2
    assert summary.intent_accuracy == 1.0
    assert summary.invented_detail_rate == 0.5
    assert summary.placeholder_rate == 0.5
    assert rows[0]["invented_details"] == {"phone": ["1-800-555-0199"]}
    assert set(summary.as_dict()) == {
        "n",
        "intent_accuracy",
        "rouge_l",
        "invented_detail_rate",
        "placeholder_rate",
        "avg_words",
    }
