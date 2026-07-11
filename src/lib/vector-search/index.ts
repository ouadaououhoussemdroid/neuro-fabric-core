import { cosine, l2 } from "./cosine";

export { cosine, l2 };

export interface IndexedVector<M = unknown> {
  id: string;
  vector: number[];
  meta?: M;
}

export interface SearchHit<M = unknown> {
  id: string;
  score: number;
  meta?: M;
}

export class VectorIndex<M = unknown> {
  private items: IndexedVector<M>[] = [];

  add(item: IndexedVector<M>): void {
    this.items.push(item);
  }

  addAll(items: IndexedVector<M>[]): void {
    for (const it of items) this.items.push(it);
  }

  size(): number {
    return this.items.length;
  }

  /** Brute-force top-k cosine similarity search. */
  search(query: number[], k = 8): SearchHit<M>[] {
    const hits: SearchHit<M>[] = this.items.map((it) => ({
      id: it.id,
      score: cosine(query, it.vector),
      meta: it.meta,
    }));
    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, k);
  }

  /** Single nearest neighbour. */
  nearest(query: number[]): SearchHit<M> | null {
    if (this.items.length === 0) return null;
    return this.search(query, 1)[0];
  }
}
