import { describe, it, expect, beforeEach } from "vitest";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

beforeEach(() => {
  process.env.DASHBOARD_DATA_DIR = mkdtempSync(path.join(tmpdir(), "post-export-test-"));
});

// Same columns and quoting as Meta's per-post exports, including multi-line captions.
const FACEBOOK = `"Post ID","Page ID","Page name",Title,Description,"Duration (sec)","Publish time","Caption type",Permalink,"Is crosspost","Is share","Post type",Languages,"Custom labels","Funded content status","Data comment",Date,Views,Reach,"Reactions, Comments and Shares",Reactions,Comments,Shares,"Total clicks","Other Clicks","Matched Audience Targeting Consumption (Photo Click)"
1827720132094663,100045700821521,"Studiostone Creative","An autumn favourite is back! 🍁 
Cooler days are a good excuse to slow down, ""really"".

#StoneCarving #MapleLeaf",,0,"09/28/2026 18:12",N/A,https://www.facebook.com/studiostonecreative/posts/pfbidNEW1,0,0,Photos,,,,,Lifetime,27,20,4,4,0,0,2,1,1
1823209189212424,100045700821521,"Studiostone Creative","Take yourself back to the shore. 🤍 
The sound of the water.",,0,"09/24/2026 15:30",N/A,https://www.facebook.com/studiostonecreative/posts/pfbidCHANGED,0,0,Photos,,,,,Lifetime,63,45,6,5,0,1,2,1,1
`;

const INSTAGRAM = `"Post ID","Account ID","Account username","Account name",Description,"Duration (sec)","Publish time",Permalink,"Post type","Data comment",Date,Views,Reach,Likes,Shares,Follows,Comments,Saves
18182686576415017,17841403398239763,studiostonecreative,"Studiostone Creative","Take yourself back to the shore. 🤍 
The sound of the water.",0,"09/24/2026 15:30",https://www.instagram.com/p/Ddr9jGEmLMp/,"IG carousel",,Lifetime,99,40,4,0,0,0,0
`;

describe("Meta per-post exports", () => {
  it("parses each platform's layout, keeping multi-line captions whole", async () => {
    const { parseMetaPostExport } = await import("@/lib/marketing/socialInsightsImport");
    const fb = parseMetaPostExport(FACEBOOK)!;
    expect(fb).toHaveLength(2);
    expect(fb[0]).toMatchObject({ platform: "facebook", externalId: "1827720132094663", postedDate: "2026-09-28", postType: "Photos", views: 27, reach: 20, likes: 4, comments: 0, shares: 0, linkClicks: 1 });
    expect(fb[0]!.caption).toBe('An autumn favourite is back! 🍁 \nCooler days are a good excuse to slow down, "really".\n\n#StoneCarving #MapleLeaf');
    const ig = parseMetaPostExport(INSTAGRAM)!;
    expect(ig[0]).toMatchObject({ platform: "instagram", postedDate: "2026-09-24", views: 99, reach: 40, likes: 4, linkClicks: null });
    expect(parseMetaPostExport("platform,record_type,metric\nFacebook,summary,Views")).toBeNull();
  });

  it("refreshes posts already in the Post log and adds new ones, and re-importing changes nothing", async () => {
    const { createSocialPost, listSocialPosts } = await import("@/lib/marketing/socialPosts");
    const { importSocialInsightsCsv } = await import("@/lib/marketing/socialInsightsImport");
    // Imported earlier, before Post IDs were stored, under a permalink Facebook has since changed.
    const oldFb = createSocialPost({ posted_date: "2026-09-24", platform: "facebook", post_type: "Photos", caption: "Take yourself back to the shore. 🤍 \nThe sound of the water.", likes: 4, comments: 0, shares: 1, link_clicks: 1, reach: 22, notes: "Imported earlier (real, Alabaster Shell concept)." });
    const oldIg = createSocialPost({ posted_date: "2026-09-24", platform: "instagram", caption: "Take yourself back to the shore. 🤍 \nThe sound of the water.", likes: 3, comments: 0, shares: 0, link_clicks: 2, reach: 27 });

    expect(importSocialInsightsCsv(FACEBOOK)).toMatchObject({ postsImported: 1, postsUpdated: 1 });
    expect(importSocialInsightsCsv(INSTAGRAM)).toMatchObject({ postsImported: 0, postsUpdated: 1 });

    const posts = listSocialPosts();
    expect(posts).toHaveLength(3);
    const fb = posts.find((p) => p.id === oldFb.id)!;
    expect(fb).toMatchObject({ likes: 5, shares: 1, reach: 45, views: 63, link_clicks: 1, external_id: "1823209189212424", notes: "Imported earlier (real, Alabaster Shell concept)." });
    const ig = posts.find((p) => p.id === oldIg.id)!;
    expect(ig).toMatchObject({ likes: 4, reach: 40, views: 99, link_clicks: 2 }); // Instagram's file has no clicks, so they're kept
    expect(posts.find((p) => p.posted_date === "2026-09-28")).toMatchObject({ platform: "facebook", likes: 4, reach: 20, views: 27, external_id: "1827720132094663" });

    expect(importSocialInsightsCsv(FACEBOOK)).toMatchObject({ postsImported: 0, postsUpdated: 2 });
    expect(listSocialPosts()).toHaveLength(3);
  });

  it("fills in the entry the planner logged for that day instead of duplicating it", async () => {
    const { createSocialPost, listSocialPosts, PLANNER_LOG_NOTE } = await import("@/lib/marketing/socialPosts");
    const { importSocialInsightsCsv } = await import("@/lib/marketing/socialInsightsImport");
    const planned = createSocialPost({ posted_date: "2026-09-28", platform: "facebook", caption: "Maple leaf, planner wording", likes: 0, comments: 0, shares: 0, link_clicks: 0, notes: PLANNER_LOG_NOTE });
    importSocialInsightsCsv(FACEBOOK);
    const row = listSocialPosts().find((p) => p.id === planned.id)!;
    expect(row.likes).toBe(4);
    expect(row.caption).toContain("An autumn favourite is back!");
    expect(row.notes).toBe("Logged from the planner; numbers from a Meta post export.");
  });
});
