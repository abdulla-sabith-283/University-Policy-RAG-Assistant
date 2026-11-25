from src.retriever import Retriever


TEST_CASES = [
    # ---------------------------------------------------------
    # ANSWERABLE
    # ---------------------------------------------------------

    {
        "question": "What is the minimum attendance requirement?",
        "expected_sources": [
            "NB-AL-03_Attendance_and_Leave_Policy.pdf",
            "NB-AR-01_Academic_Regulations_and_Progression_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": (
            "How much attendance does a student need to be eligible "
            "for an end-semester examination?"
        ),
        "expected_sources": [
            "NB-AR-01_Academic_Regulations_and_Progression_Policy.pdf",
            "NB-EX-02_Examination_and_Assessment_Regulations.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "How many items can an undergraduate borrow from the library?",
        "expected_sources": [
            "NB-LB-05_Library_Services_and_Information_Resources_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "Can an approved internship affect a student's attendance?",
        "expected_sources": [
            "NB-IN-04_Internship_and_Experiential_Learning_Policy.pdf",
            "NB-AL-03_Attendance_and_Leave_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "Can an approved internship waive examination requirements?",
        "expected_sources": [
            "NB-IN-04_Internship_and_Experiential_Learning_Policy.pdf",
            "NB-EX-02_Examination_and_Assessment_Regulations.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "What happens if a student is placed on academic probation?",
        "expected_sources": [
            "NB-SC-06_Scholarship_and_Financial_Assistance_Policy.pdf",
            "NB-IN-04_Internship_and_Experiential_Learning_Policy.pdf",
            "NB-AR-01_Academic_Regulations_and_Progression_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "Does meeting the CGPA requirement guarantee scholarship renewal?",
        "expected_sources": [
            "NB-SC-06_Scholarship_and_Financial_Assistance_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "What is the loan period for an undergraduate library item?",
        "expected_sources": [
            "NB-LB-05_Library_Services_and_Information_Resources_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "Who approves an overload request?",
        "expected_sources": [
            "NB-AR-01_Academic_Regulations_and_Progression_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "How long before an internship must approval be obtained?",
        "expected_sources": [
            "NB-IN-04_Internship_and_Experiential_Learning_Policy.pdf",
            "NB-AL-03_Attendance_and_Leave_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "What happens when attendance falls below 75%?",
        "expected_sources": [
            "NB-AL-03_Attendance_and_Leave_Policy.pdf",
        ],
        "answerable": True,
    },

    {
        "question": "What identification must students carry during examinations?",
        "expected_sources": [
            "NB-EX-02_Examination_and_Assessment_Regulations.pdf",
        ],
        "answerable": True,
    },

    # ---------------------------------------------------------
    # UNANSWERABLE
    # ---------------------------------------------------------

    {
        "question": "What is the hostel curfew?",
        "expected_sources": [],
        "answerable": False,
    },

    {
        "question": "What are the hostel dining hours?",
        "expected_sources": [],
        "answerable": False,
    },

    {
        "question": "How many parking spaces are allocated to students?",
        "expected_sources": [],
        "answerable": False,
    },

    {
        "question": "What is the campus gym membership fee?",
        "expected_sources": [],
        "answerable": False,
    },

    {
        "question": "What time does the hostel close?",
        "expected_sources": [],
        "answerable": False,
    },
]


def evaluate():

    retriever = Retriever()

    answerable_total = 0
    answerable_passed = 0

    unanswerable_total = 0
    unanswerable_rejected = 0

    total_source_coverage = 0.0

    for i, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected_sources = set(test["expected_sources"])
        answerable = test["answerable"]

        result = retriever.retrieve_with_gate(
            question,
            top_k=5,
        )

        results = result["results"]
        relevant = result["relevant"]

        retrieved_sources = {
            item["metadata"]["source"]
            for item in results
        }

        found_sources = (
            expected_sources.intersection(retrieved_sources)
        )

        if answerable:

            answerable_total += 1

            coverage = (
                len(found_sources)
                / len(expected_sources)
            )

            total_source_coverage += coverage

            if coverage == 1.0 and relevant:
                answerable_passed += 1

            status = (
                "PASS"
                if coverage == 1.0 and relevant
                else "FAIL"
            )

        else:

            unanswerable_total += 1

            if not relevant:
                unanswerable_rejected += 1
                status = "PASS"
            else:
                status = "FAIL"

            coverage = None

        print("\n" + "=" * 80)
        print(f"TEST {i}")
        print("=" * 80)

        print(f"\nQuestion:")
        print(question)

        print(
            f"\nType: "
            f"{'ANSWERABLE' if answerable else 'UNANSWERABLE'}"
        )

        if results:
            print(
                f"\nBest distance: "
                f"{results[0]['distance']:.4f}"
            )

        print(
            f"Gate decision: "
            f"{'ACCEPT' if relevant else 'REJECT'}"
        )

        if answerable:

            print(
                f"Source coverage: "
                f"{coverage:.0%}"
            )

        print(f"Result: {status}")

    average_coverage = (
        total_source_coverage / answerable_total
        if answerable_total
        else 0
    )

    answerable_accuracy = (
        answerable_passed / answerable_total
        if answerable_total
        else 0
    )

    abstention_accuracy = (
        unanswerable_rejected / unanswerable_total
        if unanswerable_total
        else 0
    )

    overall_gate_accuracy = (
        (answerable_passed + unanswerable_rejected)
        / len(TEST_CASES)
    )

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"\nRelevance threshold: "
        f"{Retriever.RELEVANCE_THRESHOLD}"
    )

    print(
        f"\nAnswerable questions: "
        f"{answerable_passed}/{answerable_total}"
    )

    print(
        f"Answerable accuracy: "
        f"{answerable_accuracy:.0%}"
    )

    print(
        f"Average source coverage: "
        f"{average_coverage:.0%}"
    )

    print(
        f"\nUnanswerable questions correctly rejected: "
        f"{unanswerable_rejected}/{unanswerable_total}"
    )

    print(
        f"Abstention accuracy: "
        f"{abstention_accuracy:.0%}"
    )

    print(
        f"\nOverall gate accuracy: "
        f"{overall_gate_accuracy:.0%}"
    )


if __name__ == "__main__":
    evaluate()