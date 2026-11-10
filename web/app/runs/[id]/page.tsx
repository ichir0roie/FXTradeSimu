import RunView from "@/components/RunView";

export const dynamic = "force-dynamic";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <RunView id={Number(id)} />;
}
