import CaseView from "@/components/CaseView";

export const dynamic = "force-dynamic";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <CaseView id={decodeURIComponent(id)} />;
}
