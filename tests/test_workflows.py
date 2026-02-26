from sportvision.workflows.blocks import PossessionTrackerBlock, TeamClassifierBlock


class TestWorkflowBlocks:
    def test_team_classifier_block_metadata(self):
        block = TeamClassifierBlock()
        assert block.get_manifest()["type"] == "sportvision/team_classifier"

    def test_possession_tracker_block_metadata(self):
        block = PossessionTrackerBlock()
        assert block.get_manifest()["type"] == "sportvision/possession_tracker"
