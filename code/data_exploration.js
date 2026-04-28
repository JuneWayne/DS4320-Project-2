// explore.js — mongosh exploration of the ai_detection database
// Run: mongosh "<MONGO_URI>" --file code/explore.js

db = db.getSiblingDB("ai_detection");

// ── Collection sizes ─────────────────────────────────────────────────────────
print("\n=== Collection Sizes ===");
print("texts:  ", db.texts.countDocuments());
print("models: ", db.models.countDocuments());
print("domains:", db.domains.countDocuments());

// ── Label distribution (human vs AI) ─────────────────────────────────────────
print("\n=== Label Distribution ===");
db.texts.aggregate([
    { $group: { _id: "$label", count: { $sum: 1 } } },
    { $sort: { _id: 1 } }
]).forEach(d => print(d._id === 0 ? "Human:" : "AI:   ", d.count));

// ── Document count per domain ─────────────────────────────────────────────────
print("\n=== Documents per Domain ===");
db.texts.aggregate([
    { $group: { _id: "$domain_id", count: { $sum: 1 } } },
    { $sort: { count: -1 } }
]).forEach(d => print(d._id.padEnd(12), d.count));

// ── Document count per model ──────────────────────────────────────────────────
print("\n=== Documents per Model ===");
db.texts.aggregate([
    { $group: { _id: "$model_id", count: { $sum: 1 } } },
    { $sort: { count: -1 } }
]).forEach(d => print(d._id.padEnd(14), d.count));

// ── Average linguistic features by label ─────────────────────────────────────
print("\n=== Avg Features: Human (0) vs AI (1) ===");
db.texts.aggregate([
    {
        $group: {
            _id: "$label",
            avg_word_count: { $avg: "$word_count" },
            avg_sentence_length: { $avg: "$avg_sentence_length" },
            avg_unique_word_ratio: { $avg: "$unique_word_ratio" },
            avg_word_length: { $avg: "$avg_word_length" },
            avg_punct_density: { $avg: "$punctuation_density" }
        }
    },
    { $sort: { _id: 1 } }
]).forEach(d => {
    print(`\nLabel ${d._id === 0 ? "Human" : "AI"}:`);
    print("  word_count:        ", d.avg_word_count.toFixed(1));
    print("  sentence_length:   ", d.avg_sentence_length.toFixed(2));
    print("  unique_word_ratio: ", d.avg_unique_word_ratio.toFixed(3));
    print("  avg_word_length:   ", d.avg_word_length.toFixed(3));
    print("  punct_density:     ", d.avg_punct_density.toFixed(4));
});

// ── Sample human document ─────────────────────────────────────────────────────
print("\n=== Sample Human Document ===");
const human = db.texts.findOne({ label: 0 }, { text: 0 }); // omit full text blob
printjson(human);

// ── Sample AI document ───────────────────────────────────────────────────────
print("\n=== Sample AI Document ===");
const ai = db.texts.findOne({ label: 1 }, { text: 0 });
printjson(ai);

// ── All model documents ───────────────────────────────────────────────────────
print("\n=== Models Collection ===");
db.models.find().forEach(d => printjson(d));

// ── All domain documents ──────────────────────────────────────────────────────
print("\n=== Domains Collection ===");
db.domains.find().forEach(d => printjson(d));
