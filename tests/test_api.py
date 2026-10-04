"""极大间隙惩罚下关联重算的 API 级回归测试。

复现原始故障：独立数据中录入参考孔两层、待校核孔一层，五项相似度权重
全设为 0，`PUT /api/settings` 设置 `gap_open=1e20`、`gap_extend=1`，
不锁标志层调用 `POST /api/correlate`——设置成功但重算曾返回 500
（不可达哨兵为有限值 -1e18，合法负分 -1e20 被当作不可达，规范回溯
索引越界）。修复后应返回 200、`dp_score=-1e20`、两个并列最优归属，
且主解、并列方案与导出 JSON 对相同参数给出一致结果。

需要 Flask（Docker 镜像内已装）；本地无 Flask 时本文件整体跳过。
核心算法回归见 test_correlator.py（不依赖 Flask）。

运行：python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

ZERO_WEIGHTS = {"lithology": 0, "grain": 0, "water": 0,
                "elev": 0, "confidence": 0}

LAYER = {"lithology": "砂岩", "grain": "中粒", "water": "稍湿",
         "confidence": 0.9}


@unittest.skipUnless(HAS_FLASK, "Flask 不可用，跳过 API 级回归")
class HugeGapPenaltyApiTest(unittest.TestCase):
    """独立数据 + gap_open=1e20 的重算不再 500，且各输出口径一致。"""

    def setUp(self):
        import app.db as dbmod
        self._tmp = tempfile.mkdtemp()
        dbmod.DB_PATH = Path(self._tmp) / "test.db"
        from app.routes import create_app, app as flask_app
        create_app()
        self.client = flask_app.test_client()
        # 独立数据：清空演示钻孔与标志层
        state = self.client.get("/api/state").get_json()
        for m in state["markers"]:
            self.client.delete(f"/api/markers/{m['id']}")
        for b in state["boreholes"]:
            self.client.delete(f"/api/boreholes/{b['id']}")
        # 参考孔两层、待校核孔一层
        r = self.client.post("/api/boreholes", json={
            "name": "R", "collar_elev": 100, "x": 0,
            "layers": [dict(LAYER, top=0, bottom=2),
                       dict(LAYER, top=2, bottom=4)]})
        self.assertEqual(r.status_code, 201)
        r = self.client.post("/api/boreholes", json={
            "name": "H", "collar_elev": 100, "x": 10,
            "layers": [dict(LAYER, top=0, bottom=2)]})
        self.assertEqual(r.status_code, 201)
        # 五项权重全 0、极大开间隙惩罚
        r = self.client.put("/api/settings", json={
            "weights": ZERO_WEIGHTS, "gap_open": 1e20, "gap_extend": 1})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["gap_open"], 1e20)

    def _correlate(self):
        state = self.client.get("/api/state").get_json()
        rid = next(b["id"] for b in state["boreholes"] if b["name"] == "R")
        r = self.client.post("/api/correlate",
                             json={"reference_id": rid, "marker_ids": []})
        self.assertEqual(r.status_code, 200, r.get_json())
        holes = r.get_json()["holes"]
        self.assertEqual(len(holes), 1)
        return holes[0]

    def test_recalculate_no_500_and_two_ties(self):
        h = self._correlate()
        self.assertEqual(h["dp_score"], -1e20)
        self.assertEqual(h["solution_count"], 2)
        self.assertEqual(h["solutions_listed"], 2)
        sigs = {
            (tuple(map(tuple, s["pairs"])),
             tuple(s["unmatched_ref"]), tuple(s["unmatched_hole"]))
            for s in h["tie_solutions"]
        }
        # 目标层分别匹配参考孔第一层或第二层，各留一个未匹配参考层
        self.assertEqual(len(sigs), 2)
        for pairs, unref, unhole in sigs:
            self.assertEqual(len(pairs), 1)
            self.assertEqual(len(unref), 1)
            self.assertEqual(unhole, ())
        matched_ref = sorted(p[0] for p, _, _ in
                             [(p, u, uh) for p, u, uh in sigs] for p in p)
        unmatched_ref = sorted(u[0] for _, u, _ in sigs)
        self.assertEqual(matched_ref, unmatched_ref)  # 同一对参考层互换角色
        # 主解是两个最优归属之一
        primary = (
            tuple((p["ref_layer_id"], p["hole_layer_id"]) for p in h["pairs"]),
            tuple(u["layer_id"] for u in h["unmatched_ref"]),
            tuple(u["layer_id"] for u in h["unmatched_hole"]),
        )
        self.assertIn(primary, sigs)
        # 两个参考层均判出“可未匹配”的歧义
        self.assertEqual(len(h["ambiguities"]), 2)
        for amb in h["ambiguities"]:
            self.assertTrue(amb["can_be_unmatched"])
            self.assertEqual(amb["outcome_count"], 2)

    def test_export_and_rerun_consistent(self):
        h = self._correlate()
        # 相同参数再次重算：结果一致
        h2 = self._correlate()
        self.assertEqual(h2["dp_score"], h["dp_score"])
        self.assertEqual(h2["tie_solutions"], h["tie_solutions"])
        self.assertEqual(h2["pairs"], h["pairs"])
        # 导出 JSON 归档的最近一次结果与重算一致
        payload = json.loads(self.client.get("/api/export").data)
        last = payload["last_result"]["holes"][0]
        self.assertEqual(last["dp_score"], -1e20)
        self.assertEqual(last["solution_count"], 2)
        self.assertEqual(last["solutions_listed"], 2)
        self.assertEqual(last["tie_solutions"], h["tie_solutions"])
        self.assertEqual(last["pairs"], h["pairs"])
        self.assertEqual(payload["settings"]["gap_open"], 1e20)
        self.assertEqual(payload["last_result"]["parameters"]["gap_open"], 1e20)


if __name__ == "__main__":
    unittest.main()
