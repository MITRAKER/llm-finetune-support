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
        ("Please call our customer service at 555-0199.", "phone"),
        ("Contact our customer support team at 555-0199 for help.", "phone"),
        ("You can reach our support team at 555-0199.", "phone"),
        ("Call us 24/7 at 555-0199.", "phone"),
        ("You can also try calling 555-0199.", "phone"),
        ("Texting 555-0199 works too.", "phone"),
        ("Mobile: 555-0199", "phone"),
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


@pytest.mark.parametrize(
    "text",
    [
        "Orders of 100-1000 units ship by freight.",
        "Orders of 150-2500 ship by freight.",
        "The warehouse is open 900-1700 on weekdays.",
        "Our phone lines are open 900-1700 on weekdays.",
        "You can call us between 800-2000, seven days a week.",
        "Contact us about orders of 250-5000 units.",
        "Feel free to contact us. Orders of 250-5000 ship by freight.",
    ],
)
def test_number_ranges_are_not_phone_numbers(text):
    """A short number needs a word such as "call" before it in the same sentence, and even
    then a range of round hundreds or a number followed by a unit is not a phone number."""
    assert "phone" not in m.invented_details(text)


def test_a_short_number_after_a_word_such_as_call_is_a_phone_number():
    """Only the number is reported, not the word that marks it as a phone number."""
    assert m.invented_details("Call us at 555 0199 for help.") == {"phone": ["555 0199"]}
    assert m.invented_details("Our helpline number is 555-0199.") == {"phone": ["555-0199"]}
    assert m.invented_details("Tel. 555 0199, any time.") == {"phone": ["555 0199"]}
    # Every number after the word counts, so an invented second number is caught.
    found = m.invented_details("Call 555-0199 or 555-0123.", user_message="call me at 555-0199")
    assert found == {"phone": ["555-0123"]}
    # A short number inside a longer one is not reported twice.
    assert m.invented_details("Call us at (555) 123-4567.") == {"phone": ["(555) 123-4567"]}


@pytest.mark.parametrize(
    ("reply", "user"),
    [
        ("We will call you at 555-0199.", "My new mobile is 555-0199, please update it."),
        ("We'll text 555 0199.", "My cell is 555 0199"),
        ("Your order number 482-9137 is on its way.", "Where is my order 482-9137?"),
    ],
)
def test_a_short_number_the_customer_gave_may_be_repeated(reply, user):
    """The customer's own short number needs no word such as "call" to count as given."""
    assert m.invented_details(reply, user_message=user) == {}


def test_a_phone_number_the_customer_gave_may_gain_or_lose_its_country_code():
    assert m.invented_details("We will call +1 555 123 4567 today.", user_message="Call me on 555-123-4567") == {}
    assert m.invented_details("We will call +44 20 7946 0958.", user_message="call me on 020 7946 0958") == {}
    assert m.invented_details("We will call +49 30 1234 5678.", user_message="call me on 030 1234 5678") == {}
    found = m.invented_details("We will call +1 555 123 4568.", user_message="Call me on 555-123-4567")
    assert found == {"phone": ["+1 555 123 4568"]}


@pytest.mark.parametrize(
    ("reply", "user"),
    [
        ("We'll call you back at (212) 555-0199.", "call me at 555-0199"),
        ("Call us at 1-800-555-0199.", "my number is 555-0199"),
        ("Call us at 1-800-123-4567.", "My order number is 1234567."),
        ("Call +1 800 4567 1234 for help.", "My order 45671234 is late"),
        ("Call us at +1 800 555 0199.", "My tracking number is 94001118005550199"),
        ("Call us at 555-0199.", "Order 0199 is late"),
    ],
)
def test_only_a_country_code_may_be_added_to_a_number_the_customer_gave(reply, user):
    """An invented area code, or an invented number that ends in the customer's order
    number, is not the customer's phone number."""
    assert "phone" in m.invented_details(reply, user_message=user)


def test_the_country_code_rule_is_for_phone_numbers_only():
    found = m.invented_details("Your new order number is 9912345678.", user_message="Where is order 12345678?")
    assert found == {"long_number": ["9912345678"]}


def test_a_link_ends_before_brackets_and_punctuation():
    given = "Is https://shop.example.com/track the right page?"
    assert m.invented_details("Yes: [track it](https://shop.example.com/track).", user_message=given) == {}
    assert m.invented_details("Yes, https://shop.example.com/track, as you said.", user_message=given) == {}
    assert m.invented_details("Yes: **https://shop.example.com/track**", user_message=given) == {}
    assert m.invented_details("Yes: \u201chttps://shop.example.com/track\u201d", user_message=given) == {}
    assert m.invented_details("Yes, https://shop.example.com/track\u2026", user_message=given) == {}
    found = m.invented_details("See [returns](https://shop.example.com/returns).")
    assert found == {"url": ["https://shop.example.com/returns"]}


@pytest.mark.parametrize(
    ("reply", "user"),
    [
        ("We will refund $100 today.", "I paid 100 for it."),
        ("We will refund $100 today.", "I was charged 100 for it"),
        ("Your payment of $50 was declined.", "My payment of 50 was declined"),
        ("I can see the two charges of $25 on your card.", "There are two charges of 25 on my card"),
        ("We will refund $1,299.", "My payment of 1,299 bounced"),
        ("We will refund $120.", "I bought it for 120 last week"),
        ("We will refund $100 today.", "I want my 100 refunded"),
    ],
)
def test_a_bare_number_next_to_a_money_word_excuses_the_amount(reply, user):
    assert m.invented_details(reply, user_message=user) == {}


@pytest.mark.parametrize(
    "user",
    [
        "Order 100 never arrived.",
        "Order 100 sent to Europe never arrived.",  # "Eur" in "Europe" is not a money word
        "Order 100 never arrived and I want a refund.",  # the money word is too far away
    ],
)
def test_a_bare_number_far_from_a_money_word_does_not_excuse_an_amount(user):
    assert m.invented_details("We will refund $100 today.", user_message=user) == {"money": ["$100"]}


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
